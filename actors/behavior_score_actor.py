import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import numpy as np
import torch
from monarch.actor import Actor, endpoint

from .utils import (
    chunked_with_bounds,
    count_negative_prompts,
    ensure_pad_token,
    find_block_list,
    load_contexts_for_concept,
    load_model_and_tokenizer,
    load_steer_vector,
    maybe_apply_chat_template,
    model_slug,
    one_token_ids,
    set_left_padding,
)


# -----------------------------
# Config (sent as dict)
# -----------------------------


@dataclass
class BehaviorConfig:
    # model loading
    generator_dtype: str = "bfloat16"
    judge_dtype: str = "bfloat16"

    judge_model_name: str = ""  # required by runner

    seed: int = 0

    # alpha grid
    alpha_start: float = -40.0
    alpha_end: float = 40.0
    alpha_steps: int = 41

    # steering
    normalize: bool = False
    apply_last_token_only: bool = False

    # generator sampling
    n_samples_per_context: int = 12
    gen_context_batch_size: int = 12  # number of distinct contexts per generate() call

    max_prompt_length: int = 512
    max_new_tokens: int = 100
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95

    # optional suffix appended to every context prompt before generation
    generator_prompt_suffix: str = ""

    # judge
    judge_use_chat_template: bool = True
    judge_system_prompt: str = "You are a classifier. Answer strictly with 1 or 0."
    judge_question_template: str = (
        "Text:\\n{completion}\\n\\n"
        "Does the text above express the concept {concept} (any word linked to the concept in the whole sentence even if it do not make sense is sufficient for a positive reply)? Reply with 1 for yes and 0 for no.\\n"
        "Answer:"
    )
    judge_max_prompt_length: int = 4000
    judge_batch_size: int = 64
    judge_max_completion_chars: int = 4000

    # progress / responsiveness
    progress_every: int = 1


# -----------------------------
# Actor (one GPU)
# -----------------------------


class BehaviorActor(Actor):
    """One actor per GPU.

    Caches:
      - generator model/tokenizer
      - judge model/tokenizer

    and computes p(concept present) curves vs alpha.
    """

    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True

        # generator cache
        self._gen_name: Optional[str] = None
        self._gen_dtype: Optional[str] = None
        self._gen_tok = None
        self._gen_model = None

        # judge cache
        self._judge_name: Optional[str] = None
        self._judge_dtype: Optional[str] = None
        self._judge_tok = None
        self._judge_model = None

        # cached judge candidate token ids for "1" and "0"
        self._judge_token_ids_10: Optional[Tuple[List[int], List[int]]] = None

    # --------- model management ---------

    def _ensure_generator(self, model_name: str, dtype_str: str):
        if self._gen_model is not None and self._gen_name == model_name and self._gen_dtype == dtype_str:
            return
        self._gen_tok = None
        self._gen_model = None
        torch.cuda.empty_cache()
        self._gen_tok, self._gen_model = load_model_and_tokenizer(model_name, dtype_str)
        set_left_padding(self._gen_tok)
        ensure_pad_token(self._gen_tok, self._gen_model)
        self._gen_name = model_name
        self._gen_dtype = dtype_str

    def _ensure_judge(self, model_name: str, dtype_str: str):
        if self._judge_model is not None and self._judge_name == model_name and self._judge_dtype == dtype_str:
            return
        self._judge_tok = None
        self._judge_model = None
        self._judge_token_ids_10 = None
        torch.cuda.empty_cache()
        self._judge_tok, self._judge_model = load_model_and_tokenizer(model_name, dtype_str)
        set_left_padding(self._judge_tok)
        ensure_pad_token(self._judge_tok, self._judge_model)
        self._judge_name = model_name
        self._judge_dtype = dtype_str

    # --------- judge scoring ---------

    def _ensure_judge_token_ids(self):
        assert self._judge_tok is not None
        if self._judge_token_ids_10 is not None:
            return
        tok = self._judge_tok
        # include whitespace/newline variants for robustness
        ones = one_token_ids(tok, ["1", " 1", "\n1", "\n 1"])  # noqa: W605
        zeros = one_token_ids(tok, ["0", " 0", "\n0", "\n 0"])  # noqa: W605
        if not ones or not zeros:
            # digits should basically always work, but keep an explicit error for clarity
            raise RuntimeError(
                f"Could not find 1-token ids for judge answers. ones={ones}, zeros={zeros}. "
                "Try disabling chat template or changing the judge prompt to end with 'Answer:' (no trailing space)."
            )
        self._judge_token_ids_10 = (ones, zeros)

    def _join_samples_for_judge(self, samples: List[str], cfg: BehaviorConfig) -> str:
        """
        Join multiple samples into one mega-text for an ANY-of-K judge.
        This is where you control truncation to avoid blowing up the judge prompt.
        """
        # Optional: cap each sample so early samples don't eat the whole context window
        per_sample_cap = 800  # adjust as needed
        cleaned = []
        for s in samples:
            s = (s or "").strip()
            if per_sample_cap and len(s) > per_sample_cap:
                s = s[:per_sample_cap] + "…"
            cleaned.append(s)
    
        return "\n\n--- SAMPLE ---\n\n".join(cleaned)
    
    
    def _judge_any_batch(self, samples: List[str], concept: str, cfg: BehaviorConfig) -> torch.Tensor:
        """
        B=1 judge: takes a list of texts, concatenates them, asks once:
          "is concept present anywhere?"
        Returns a tensor shape [1] with 0.0 or 1.0.
    
        Uses regex parsing on decoded answer (tokenizer-robust).
        """
        assert self._judge_tok is not None and self._judge_model is not None
        tok = self._judge_tok
        model = self._judge_model
    
        mega_text = self._join_samples_for_judge(samples, cfg)
    
        user = cfg.judge_question_template.format(concept=concept, completion=mega_text)
        prompt = maybe_apply_chat_template(tok, cfg.judge_system_prompt, user, cfg.judge_use_chat_template)
    
        enc = tok(
            [prompt],
            return_tensors="pt",
            padding=False,          # B=1 -> no padding needed
            truncation=True,
            max_length=cfg.judge_max_prompt_length,
        )
        input_ids = enc["input_ids"].to("cuda", non_blocking=True)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(input_ids.device, non_blocking=True)
    
        input_len = int(input_ids.shape[1])
    
        pad_id = tok.pad_token_id or tok.eos_token_id or tok.bos_token_id or 0
    
        # Generate a few tokens so leading newline/space doesn't break parsing
        gen_kwargs = dict(
            max_new_tokens=4,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,   # silence "ignored top_p/top_k" warnings if present
            top_k=0,
            use_cache=True,
            pad_token_id=int(pad_id),
        )
        if tok.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = int(tok.eos_token_id)
    
        amp_judge = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with torch.inference_mode(), amp_judge:
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
    
        # Decode ONLY the newly generated tokens (critical!)
        gen_ids = out_ids[:, input_len:]  # [1, <=4]
        ans = tok.batch_decode(gen_ids, skip_special_tokens=True)[0]
    
        # Parse first 0/1
        m = re.match(r"^\s*([01])", ans)
        y = 1.0 if (m and m.group(1) == "1") else 0.0
    
        return torch.tensor([y], device=input_ids.device, dtype=torch.float32)


    # --------- generation ---------

    def _generate_completions(self, prompts: List[str], cfg: BehaviorConfig) -> List[List[str]]:
        """Generate n_samples_per_context completions for each prompt.

        Returns: list of length len(prompts), each is list[str] of length K.
        """
        assert self._gen_tok is not None and self._gen_model is not None
        tok = self._gen_tok
        model = self._gen_model

        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_length,
        )
        input_ids = enc["input_ids"].to("cuda", non_blocking=True)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(input_ids.device, non_blocking=True)

        input_len = int(input_ids.shape[1])  # padded length; safe with LEFT padding

        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = tok.eos_token_id
        if pad_id is None:
            pad_id = tok.bos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define at least one of PAD/EOS/BOS token ids.")

        gen_kwargs = dict(
            max_new_tokens=int(cfg.max_new_tokens),
            do_sample=True,
            temperature=float(cfg.temperature),
            top_k=int(cfg.top_k),
            top_p=float(cfg.top_p),
            num_return_sequences=int(cfg.n_samples_per_context),
            use_cache=True,
            pad_token_id=int(pad_id),
        )
        if tok.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = int(tok.eos_token_id)

        with torch.inference_mode():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # New tokens start after the *padded* prompt length.
        new_ids = out_ids[:, input_len:]
        texts = tok.batch_decode(new_ids, skip_special_tokens=True)

        K = int(cfg.n_samples_per_context)
        B = len(prompts)
        if len(texts) != B * K:
            raise RuntimeError(f"Unexpected generate() output size: got {len(texts)}, expected {B}*{K}={B*K}.")

        # HF repeats each input row K times via repeat_interleave => grouped by prompt.
        grouped: List[List[str]] = []
        idx = 0
        for _ in range(B):
            grouped.append([t.strip() for t in texts[idx : idx + K]])
            idx += K
        return grouped

    # --------- steering hook ---------

    @staticmethod
    def _make_steer_hook(alpha: float, steer_vec: torch.Tensor, last_token_only: bool):
        """Create a forward hook that adds alpha*steer_vec to the block output."""
        a = float(alpha)

        def _hook(module, inputs, output):
            x = output[0] if isinstance(output, (tuple, list)) else output
            # x: [B,T,H]
            if not torch.is_tensor(x):
                return output

            add = (steer_vec * a).to(dtype=x.dtype, device=x.device)  # [H]

            if last_token_only:
                # only last token position
                x2 = x.clone()
                x2[:, -1, :] = x2[:, -1, :] + add
            else:
                x2 = x + add.view(1, 1, -1)

            if isinstance(output, (tuple, list)):
                out = list(output)
                out[0] = x2
                return tuple(out)
            return x2

        return _hook

    # -----------------------------
    # Endpoint
    # -----------------------------

    @endpoint
    async def compute_behavior_curves(
        self,
        model_name: str,
        concept_slug: str,
        concept_label: str,
        block_idx_to_steer,  # int or None for all
        contexts_file: str,
        steer_dir: str,
        save_dir: str,
        layer_path: Optional[str] = None,
        cfg_dict: Optional[Dict[str, Any]] = None,
        rank_hint: int = 0,
    ) -> Dict[str, Any]:
        cfg = BehaviorConfig(**(cfg_dict or {}))
        if not cfg.judge_model_name:
            raise ValueError("BehaviorConfig.judge_model_name must be set (via --judge_model).")

        # deterministic seed per GPU
        torch.manual_seed(int(cfg.seed) + int(rank_hint))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed) + int(rank_hint))

        # load models
        self._ensure_generator(model_name, cfg.generator_dtype)
        self._ensure_judge(cfg.judge_model_name, cfg.judge_dtype)
        gen_tok, gen_model = self._gen_tok, self._gen_model
        assert gen_tok is not None and gen_model is not None

        # resolve blocks
        blocks = find_block_list(gen_model, override_path=layer_path)
        n_blocks = len(blocks)
        if block_idx_to_steer is None:
            layer_indices = list(range(n_blocks))
        else:
            # Interpret integer input as "sample this many layers uniformly across depth".
            layer_indices = np.linspace(0, n_blocks - 1, num=int(block_idx_to_steer), dtype=int).tolist()
        if not layer_indices:
            return {"error": "No valid layer indices."}

        # contexts for this concept
        try:
            contexts, context_source_lines = load_contexts_for_concept(
                contexts_file,
                concept_slug=concept_slug,
                concept_label=concept_label,
            )
        except Exception as e:
            return {"error": f"Failed to load contexts for concept '{concept_slug}': {e}"}

        if not contexts:
            return {"error": f"No contexts in {contexts_file} for concept '{concept_slug}'"}

        n_neg = count_negative_prompts(contexts_file)
        if n_neg is None:
            ctx_is_positive = np.full((len(contexts),), -1, dtype=np.int8)
        else:
            if n_neg > len(contexts):
                n_neg = len(contexts)
            ctx_is_positive = np.zeros((len(contexts),), dtype=np.int8)
            ctx_is_positive[n_neg:] = 1

        # alpha grid (ensure 0 present)
        alphas = torch.linspace(cfg.alpha_start, cfg.alpha_end, steps=int(cfg.alpha_steps), dtype=torch.float32)
        if not bool((alphas == 0.0).any()):
            alphas = torch.sort(torch.cat([alphas, torch.tensor([0.0], dtype=torch.float32)]))[0]
        alphas_np = alphas.cpu().numpy().astype(np.float32)

        # save root
        save_root = Path(save_dir) / model_slug(model_name) / concept_slug
        save_root.mkdir(parents=True, exist_ok=True)

        steer_dir_path = Path(steer_dir)
        results: List[Dict[str, Any]] = []

        # main: per layer
        for li, layer_idx in enumerate(layer_indices):
            # load steering vector
            steer_vec_cpu = load_steer_vector(steer_dir_path, model_name, concept_slug, int(layer_idx))
            if cfg.normalize:
                steer_vec_cpu = steer_vec_cpu / torch.norm(steer_vec_cpu).clamp_min(1e-8)
            steer_vec = steer_vec_cpu.to("cuda", non_blocking=True).to(torch.float32)

            # storage: p1_by_ctx[C,A]
            C = len(contexts)
            A = int(alphas.numel())
            p1_by_ctx = np.full((C, A), np.nan, dtype=np.float32)

            # for each alpha
            for ai, alpha in enumerate(alphas.tolist()):
                # register hook
                hook = self._make_steer_hook(alpha=float(alpha), steer_vec=steer_vec, last_token_only=cfg.apply_last_token_only)
                handle = blocks[int(layer_idx)].register_forward_hook(hook)

                try:
                    # iterate contexts in batches
                    for start, end, ctx_batch in chunked_with_bounds(contexts, int(cfg.gen_context_batch_size)):
                        # generator prompts
                        prompts: List[str] = []
                        for ctx in ctx_batch:
                            base = (ctx or "").rstrip()
                            if cfg.generator_prompt_suffix:
                                base = base + cfg.generator_prompt_suffix
                            prompts.append(base)


                        grouped_completions = self._generate_completions(prompts, cfg)  # List[List[str]] length B, each length K
                        
                        # Judge each context with ONE prompt (B=1 each time)
                        p1_context = []
                        for samples in grouped_completions:
                            # Any-of-K judge: one binary decision for the grouped samples of this context.
                            p1 = self._judge_any_batch(samples, concept=concept_label, cfg=cfg)  # [1]
                            p1_context.append(p1)
                        
                        p1_context = torch.cat(p1_context, dim=0)  # [B] values 0/1
                        p1_mean = p1_context.cpu().numpy().astype(np.float32)
                        p1_by_ctx[start:end, ai] = p1_mean

                        # keep mailbox responsive
                        if cfg.progress_every and ((start // max(1, int(cfg.gen_context_batch_size))) % int(cfg.progress_every) == 0):
                            await asyncio.sleep(0)

                finally:
                    handle.remove()

                if cfg.progress_every and (ai % max(1, int(cfg.progress_every)) == 0):
                    await asyncio.sleep(0)

            # summaries
            mean_all = np.nanmean(p1_by_ctx, axis=0).astype(np.float32)

            if (ctx_is_positive >= 0).all():
                neg = p1_by_ctx[ctx_is_positive == 0]
                pos = p1_by_ctx[ctx_is_positive == 1]
                mean_negative = np.nanmean(neg, axis=0).astype(np.float32) if neg.size else np.full((A,), np.nan, np.float32)
                mean_positive = np.nanmean(pos, axis=0).astype(np.float32) if pos.size else np.full((A,), np.nan, np.float32)

                # correctness: positives want 1, negatives want 0
                match = np.empty_like(p1_by_ctx)
                match[ctx_is_positive == 1] = p1_by_ctx[ctx_is_positive == 1]
                match[ctx_is_positive == 0] = 1.0 - p1_by_ctx[ctx_is_positive == 0]
                mean_match = np.nanmean(match, axis=0).astype(np.float32)
            else:
                mean_negative = np.full((A,), np.nan, dtype=np.float32)
                mean_positive = np.full((A,), np.nan, dtype=np.float32)
                mean_match = np.full((A,), np.nan, dtype=np.float32)

            # write file
            out_path = save_root / f"layer_{int(layer_idx)}_behavior.npz"
            meta = {
                "model": model_name,
                "concept_slug": concept_slug,
                "concept": concept_label,
                "judge_model": cfg.judge_model_name,
                "layer_idx": int(layer_idx),
                "alphas": {"start": float(alphas_np[0]), "end": float(alphas_np[-1]), "steps": int(A)},
                "n_contexts": int(C),
                "n_samples_per_context": int(cfg.n_samples_per_context),
                "generator": {
                    "max_prompt_length": int(cfg.max_prompt_length),
                    "max_new_tokens": int(cfg.max_new_tokens),
                    "temperature": float(cfg.temperature),
                    "top_k": int(cfg.top_k),
                    "top_p": float(cfg.top_p),
                },
                "judge": {
                    "use_chat_template": bool(cfg.judge_use_chat_template),
                    "max_prompt_length": int(cfg.judge_max_prompt_length),
                },
                "contexts_file": str(contexts_file),
            }

            np.savez_compressed(
                out_path,
                alphas=alphas_np,
                p1_by_ctx=p1_by_ctx,
                ctx_texts=np.array(contexts, dtype=object),
                ctx_source_lines=np.array(context_source_lines, dtype=np.int32),
                ctx_is_positive=ctx_is_positive,
                mean_all=mean_all,
                mean_negative=mean_negative,
                mean_positive=mean_positive,
                mean_match=mean_match,
                meta=json.dumps(meta),
            )

            results.append({"layer_idx": int(layer_idx), "file": str(out_path)})

            # allow mailbox / scheduler responsiveness between layers
            await asyncio.sleep(0)

        torch.cuda.empty_cache()
        return {"ok": True, "results": results}
