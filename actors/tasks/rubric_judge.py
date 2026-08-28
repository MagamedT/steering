"""Score whether generated text expresses a concept."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import re
from typing import Any, Sequence

import torch

from .next_token_probs import ensure_full_vocab_logits

SINGLE_ITEM_RUBRIC_JUDGE_PROMPT = """
Decide whether the single rubric item is met by the assistant response.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

Return only one JSON object with exactly these two keys:
{"explanation_1":"brief reason","criteria_met_1":true}
Use false instead of true when the item is not met. Do not add other keys.
""".strip()

# Compatibility name retained for callers that imported the previous constant.
OFFICIAL_RUBRIC_JUDGE_PROMPT = SINGLE_ITEM_RUBRIC_JUDGE_PROMPT


DEFAULT_CONCEPT_RUBRIC = (
    "1. Decide only whether the assistant response contains lexical or semantic "
    "evidence of the concept {concept}. Repeating the word {concept}, a synonym, "
    "or closely related concept words counts as present, even when the response is "
    "incoherent, nonsensical, repetitive, low quality, or unrelated to the user "
    "request. Do not judge writing quality, helpfulness, correctness, coherence, "
    "instruction following, or contextual relevance. Use only the assistant "
    "response as evidence. [Hard Rule]"
)


_VERDICT_RE = re.compile(
    r'"criteria_met_1"\s*:\s*(true|false)(?![A-Za-z0-9_])'
)


@dataclass(frozen=True)
class RubricJudgeDetails:
    """Scores and details from the judge output."""

    scores: torch.Tensor
    raw_json: tuple[str, ...]
    criteria_met: tuple[bool, ...]
    true_logits: torch.Tensor
    false_logits: torch.Tensor
    pair_mass: torch.Tensor
    verdict_token_ids: tuple[int, ...]
    true_token_ids: tuple[int, ...]
    false_token_ids: tuple[int, ...]
    verdict_positions: tuple[int, ...]


@dataclass(frozen=True)
class _BooleanTokenPair:
    """Token IDs for the judge answers true and false."""
    true_text: str
    false_text: str
    true_id: int
    false_id: int


@dataclass(frozen=True)
class _ParsedVerdict:
    """A parsed judge answer and its token position."""
    raw_json: str
    criteria_met: bool
    verdict_position: int
    verdict_token_id: int
    token_pair: _BooleanTokenPair


def pair_probability(
    true_logit: float | torch.Tensor,
    false_logit: float | torch.Tensor,
) -> float | torch.Tensor:
    """Convert true and false logits into a probability."""

    tensor_input = torch.is_tensor(true_logit) or torch.is_tensor(false_logit)
    if torch.is_tensor(true_logit):
        device = true_logit.device
    elif torch.is_tensor(false_logit):
        device = false_logit.device
    else:
        device = None
    true_value = torch.as_tensor(true_logit, dtype=torch.float32, device=device)
    false_value = torch.as_tensor(false_logit, dtype=torch.float32, device=device)
    try:
        true_value, false_value = torch.broadcast_tensors(true_value, false_value)
    except RuntimeError as exc:
        raise ValueError("true and false logits are not broadcast-compatible.") from exc
    if not bool(torch.isfinite(true_value).all() and torch.isfinite(false_value).all()):
        raise ValueError("True and false logits must be finite.")
    score = torch.sigmoid(true_value - false_value).to(torch.float32)
    if not bool(torch.isfinite(score).all()):
        raise ValueError("Pair-normalized Boolean probability is not finite.")
    if tensor_input:
        return score
    if score.numel() != 1:
        raise ValueError("Scalar logits must produce one scalar probability.")
    return float(score.item())


def build_rubric_judge_prompt(
    tokenizer,
    instruction: str,
    completion: str,
    rubric: str,
) -> str:
    """Build the prompt used to judge one completion."""

    instruction = "" if instruction is None else str(instruction)
    completion = "" if completion is None else str(completion)
    rubric = str(rubric)
    if not rubric.strip():
        raise ValueError("The rubric must be nonempty.")
    conversation = f"user: {instruction}\n\nassistant: {completion}"
    user_text = (
        SINGLE_ITEM_RUBRIC_JUDGE_PROMPT
        .replace("<<conversation>>", conversation)
        .replace("<<rubric_item>>", rubric)
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _encode_no_special(tokenizer, text: str) -> list[int]:
    """Encode text without adding tokenizer control tokens."""
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Expected one tokenized string.")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _resolve_boolean_token_pairs(tokenizer) -> tuple[_BooleanTokenPair, ...]:
    """Find single-token forms of true and false."""
    pairs: list[_BooleanTokenPair] = []
    for true_text, false_text in ((" true", " false"), ("true", "false")):
        true_ids = _encode_no_special(tokenizer, true_text)
        false_ids = _encode_no_special(tokenizer, false_text)
        if len(true_ids) != 1 or len(false_ids) != 1:
            raise ValueError(
                "Rubric Boolean labels must each tokenize to exactly one token: "
                f"{true_text!r}->{true_ids}, {false_text!r}->{false_ids}."
            )
        if true_ids[0] == false_ids[0]:
            raise ValueError("True and false labels must have distinct token ids.")
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is not None and not (
            0 <= true_ids[0] < int(vocab_size)
            and 0 <= false_ids[0] < int(vocab_size)
        ):
            raise ValueError("Boolean token id lies outside the tokenizer vocabulary.")
        pairs.append(
            _BooleanTokenPair(
                true_text=true_text,
                false_text=false_text,
                true_id=true_ids[0],
                false_id=false_ids[0],
            )
        )
    return tuple(pairs)


def _decode(tokenizer, ids: Sequence[int]) -> str:
    """Decode token IDs into text."""
    try:
        return tokenizer.decode(
            [int(value) for value in ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(
            [int(value) for value in ids], skip_special_tokens=True
        )


def _parse_generated_verdict(
    tokenizer,
    generated_ids: Sequence[int],
    token_pairs: Sequence[_BooleanTokenPair],
) -> _ParsedVerdict:
    """Parse and validate one JSON judge answer."""
    ids = [int(value) for value in generated_ids]
    raw = _decode(tokenizer, ids).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Rubric judge returned malformed or truncated JSON: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            "Rubric judge output must be one JSON object."
        )
    keys = list(parsed)
    if len(keys) < 2 or len(keys) % 2 != 0:
        raise ValueError("Rubric judge JSON must contain complete rubric pairs.")
    for item_index in range(1, len(keys) // 2 + 1):
        expected_explanation = f"explanation_{item_index}"
        expected_criterion = f"criteria_met_{item_index}"
        offset = 2 * (item_index - 1)
        if keys[offset : offset + 2] != [
            expected_explanation,
            expected_criterion,
        ]:
            raise ValueError(
                "Rubric judge JSON keys must be ordered explanation_i / "
                "criteria_met_i pairs starting at i=1."
            )
        if not isinstance(parsed[expected_explanation], str):
            raise ValueError(f"{expected_explanation} must be a JSON string.")
        if type(parsed[expected_criterion]) is not bool:
            raise ValueError(f"{expected_criterion} must be a JSON Boolean.")

    matches = list(_VERDICT_RE.finditer(raw))
    if len(matches) != 1:
        raise ValueError(
            "Rubric judge output must contain one unambiguous criteria_met_1 "
            "Boolean verdict."
        )
    match = matches[0]
    verdict_text = match.group(1)
    if (verdict_text == "true") != parsed["criteria_met_1"]:
        raise ValueError("Parsed JSON Boolean does not match the verdict text.")

    candidates: list[tuple[int, _BooleanTokenPair]] = []
    for position, token_id in enumerate(ids):
        for pair in token_pairs:
            expected_id = pair.true_id if verdict_text == "true" else pair.false_id
            if token_id != expected_id:
                continue
            before = _decode(tokenizer, ids[:position])
            through = _decode(tokenizer, ids[: position + 1])
            if not raw.startswith(before) or not raw.startswith(through):
                continue
            if len(before) <= match.start(1) and len(through) >= match.end(1):
                candidates.append((position, pair))
    if len(candidates) != 1:
        raise ValueError(
            "Could not align the unique JSON Boolean verdict to one generated "
            "true/false token boundary."
        )
    position, pair = candidates[0]
    return _ParsedVerdict(
        raw_json=raw,
        criteria_met=bool(parsed["criteria_met_1"]),
        verdict_position=position,
        verdict_token_id=ids[position],
        token_pair=pair,
    )


def _pad_id(tokenizer) -> int:
    """Choose a token ID to use for padding."""
    for value in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
    ):
        if value is not None:
            return int(value)
    raise ValueError("Judge tokenizer must define PAD, EOS, or BOS.")


def _model_device(model) -> torch.device:
    """Return the device used by the judge model."""
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError) as exc:
        raise ValueError("Could not determine the judge model device.") from exc


def _amp_context(device: torch.device, judge_dtype: str):
    """Enable mixed precision only when the device supports it."""
    if device.type == "cuda" and judge_dtype == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def rubric_completion_scores(
    tokenizer,
    model,
    samples: Sequence[str],
    concept: str,
    cfg: Any,
    *,
    instructions: Sequence[str] | None = None,
    return_details: bool = False,
    tp_mesh=None,
) -> torch.Tensor | RubricJudgeDetails:
    """Score every completion for concept presence."""

    samples = ["" if sample is None else str(sample) for sample in samples]
    if not samples:
        raise ValueError("At least one completion is required for judging.")
    if instructions is None:
        instruction_values = [""] * len(samples)
    else:
        instruction_values = ["" if value is None else str(value) for value in instructions]
        if len(instruction_values) != len(samples):
            raise ValueError("instructions must have one entry per completion.")

    batch_size = int(getattr(cfg, "judge_batch_size"))
    max_prompt_length = int(getattr(cfg, "judge_max_prompt_length"))
    configured_rubric_tokens = getattr(
        cfg, "judge_rubric_max_new_tokens", None
    )
    max_new_tokens = int(
        getattr(cfg, "judge_max_new_tokens")
        if configured_rubric_tokens is None
        else configured_rubric_tokens
    )
    if batch_size <= 0 or max_prompt_length <= 0 or max_new_tokens <= 0:
        raise ValueError("Rubric judge batch and token limits must be positive.")
    judge_dtype = str(getattr(cfg, "judge_dtype", "bfloat16")).lower()
    if judge_dtype not in {"float32", "bfloat16"}:
        raise ValueError("judge_dtype must be either 'float32' or 'bfloat16'.")
    rubric_template = str(
        getattr(cfg, "judge_rubric_template", DEFAULT_CONCEPT_RUBRIC)
    )
    try:
        rubric = rubric_template.format(concept=str(concept))
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("Invalid judge_rubric_template format string.") from exc
    if not rubric.strip():
        raise ValueError("Resolved concept rubric must be nonempty.")

    token_pairs = _resolve_boolean_token_pairs(tokenizer)
    prompts = [
        build_rubric_judge_prompt(tokenizer, instruction, sample, rubric)
        for instruction, sample in zip(instruction_values, samples)
    ]
    device = _model_device(model)
    pad_id = _pad_id(tokenizer)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    all_scores: list[torch.Tensor] = []
    all_true_logits: list[torch.Tensor] = []
    all_false_logits: list[torch.Tensor] = []
    all_pair_mass: list[torch.Tensor] = []
    raw_json: list[str] = []
    criteria_met: list[bool] = []
    verdict_token_ids: list[int] = []
    true_token_ids: list[int] = []
    false_token_ids: list[int] = []
    verdict_positions: list[int] = []

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": pad_id,
    }
    if eos_id is not None:
        generation_kwargs["eos_token_id"] = int(eos_id)

    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            unpadded_prompt_ids = [
                _encode_no_special(tokenizer, prompt) for prompt in prompt_batch
            ]
            if any(not ids for ids in unpadded_prompt_ids):
                raise ValueError("Rubric judge prompt tokenized to an empty sequence.")
            longest = max(len(ids) for ids in unpadded_prompt_ids)
            if longest > max_prompt_length:
                raise ValueError(
                    "Rubric judge prompt exceeds judge_max_prompt_length; "
                    "refusing to truncate the conversation or rubric."
                )
            prompt_width = longest
            input_ids = torch.full(
                (len(unpadded_prompt_ids), prompt_width),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, ids in enumerate(unpadded_prompt_ids):
                values = torch.tensor(ids, dtype=torch.long, device=device)
                input_ids[row, -len(ids) :] = values
                attention_mask[row, -len(ids) :] = 1

            with _amp_context(device, judge_dtype):
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
            if not torch.is_tensor(generated) or generated.ndim != 2:
                raise RuntimeError("Rubric judge generate() must return token IDs.")
            if int(generated.shape[0]) != len(prompt_batch):
                raise RuntimeError("Unexpected Rubric judge generation batch size.")
            generated_rows = generated[:, prompt_width:].detach().cpu().tolist()
            parsed_rows = [
                _parse_generated_verdict(tokenizer, ids, token_pairs)
                for ids in generated_rows
            ]
            scoring_contexts = [
                prompt_ids + generated_ids[: parsed.verdict_position]
                for prompt_ids, generated_ids, parsed in zip(
                    unpadded_prompt_ids, generated_rows, parsed_rows
                )
            ]
            context_width = max(len(ids) for ids in scoring_contexts)
            scoring_ids = torch.full(
                (len(scoring_contexts), context_width),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            scoring_mask = torch.zeros_like(scoring_ids)
            for row, ids in enumerate(scoring_contexts):
                values = torch.tensor(ids, dtype=torch.long, device=device)
                scoring_ids[row, -len(ids) :] = values
                scoring_mask[row, -len(ids) :] = 1

            with _amp_context(device, judge_dtype):
                outputs = model(
                    input_ids=scoring_ids,
                    attention_mask=scoring_mask,
                    use_cache=False,
                    logits_to_keep=1,
                )
            logits = ensure_full_vocab_logits(
                outputs.logits,
                int(model.config.vocab_size),
                tp_mesh,
            )
            if logits.ndim != 3 or logits.shape[0] != len(parsed_rows):
                raise RuntimeError("Unexpected Rubric judge logits shape.")
            final_logits = logits[:, -1, :].float()
            if not bool(torch.isfinite(final_logits).all()):
                raise ValueError("Rubric judge returned nonfinite verdict logits.")
            rows = torch.arange(len(parsed_rows), device=final_logits.device)
            true_ids = torch.tensor(
                [parsed.token_pair.true_id for parsed in parsed_rows],
                device=final_logits.device,
                dtype=torch.long,
            )
            false_ids = torch.tensor(
                [parsed.token_pair.false_id for parsed in parsed_rows],
                device=final_logits.device,
                dtype=torch.long,
            )
            z_true = final_logits[rows, true_ids]
            z_false = final_logits[rows, false_ids]
            scores = pair_probability(z_true, z_false)
            assert torch.is_tensor(scores)
            pair_log_mass = torch.logsumexp(
                torch.stack((z_true, z_false), dim=-1), dim=-1
            ) - torch.logsumexp(final_logits, dim=-1)
            pair_mass = torch.exp(pair_log_mass).clamp(0.0, 1.0)
            if not bool(torch.isfinite(pair_mass).all()):
                raise ValueError("Rubric judge pair-mass diagnostic is nonfinite.")

            all_scores.append(scores.detach().cpu())
            all_true_logits.append(z_true.detach().cpu())
            all_false_logits.append(z_false.detach().cpu())
            all_pair_mass.append(pair_mass.detach().cpu())
            raw_json.extend(parsed.raw_json for parsed in parsed_rows)
            criteria_met.extend(parsed.criteria_met for parsed in parsed_rows)
            verdict_token_ids.extend(parsed.verdict_token_id for parsed in parsed_rows)
            true_token_ids.extend(parsed.token_pair.true_id for parsed in parsed_rows)
            false_token_ids.extend(parsed.token_pair.false_id for parsed in parsed_rows)
            verdict_positions.extend(parsed.verdict_position for parsed in parsed_rows)
            # Release the scoring phase before the next generation peak.
            del (
                input_ids,
                attention_mask,
                generated,
                scoring_ids,
                scoring_mask,
                outputs,
                logits,
                final_logits,
            )

    details = RubricJudgeDetails(
        scores=torch.cat(all_scores).to(torch.float32),
        raw_json=tuple(raw_json),
        criteria_met=tuple(criteria_met),
        true_logits=torch.cat(all_true_logits).to(torch.float32),
        false_logits=torch.cat(all_false_logits).to(torch.float32),
        pair_mass=torch.cat(all_pair_mass).to(torch.float32),
        verdict_token_ids=tuple(verdict_token_ids),
        true_token_ids=tuple(true_token_ids),
        false_token_ids=tuple(false_token_ids),
        verdict_positions=tuple(verdict_positions),
    )
    if details.scores.numel() != len(samples):
        raise RuntimeError("Rubric judge did not score every completion exactly once.")
    return details if return_details else details.scores


__all__ = [
    "DEFAULT_CONCEPT_RUBRIC",
    "OFFICIAL_RUBRIC_JUDGE_PROMPT",
    "SINGLE_ITEM_RUBRIC_JUDGE_PROMPT",
    "RubricJudgeDetails",
    "build_rubric_judge_prompt",
    "pair_probability",
    "rubric_completion_scores",
]
