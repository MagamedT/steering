#!/usr/bin/env bash
set -euo pipefail

# -------------------- knobs --------------------
export PYO3_PYTHON="${PYO3_PYTHON:-/venv/main/bin/python3.12}"
export MONARCH_DIR="${MONARCH_DIR:-/workspace/monarch}"
export USE_TENSOR_ENGINE="${USE_TENSOR_ENGINE:-1}"   # default ON now
export RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-nightly}"
export TORCH_ACCELERATOR="${TORCH_ACCELERATOR:-auto}"  # auto|cpu|cu128|cu126|rocm6.3
# ------------------------------------------------

say(){ printf "\033[1;36m[+] %s\033[0m\n" "$*"; }

say "Python = $PYO3_PYTHON"
"$PYO3_PYTHON" -c 'import sys; print(sys.executable, sys.version)'

# 1) Base system libs & tools
say "Installing base system libraries"
sudo apt-get update -y
sudo apt-get install -y \
  libunwind-dev liblzma-dev liblzma5 \
  ninja-build clang pkg-config \
  rdma-core libibverbs1 ibverbs-providers \
  git curl ca-certificates
sudo ldconfig

# 2) RDMA developer headers when TE=1 (provides infiniband/mlx5dv.h)
if [ "$USE_TENSOR_ENGINE" = "1" ]; then
  say "Tensor engine enabled: installing RDMA developer packages"
  sudo apt-get install -y \
    libibverbs-dev librdmacm-dev \
    libmlx5-1

  # Fail fast if header still missing
  if [ ! -f /usr/include/infiniband/mlx5dv.h ]; then
    echo "ERROR: /usr/include/infiniband/mlx5dv.h still missing."
    echo "On non-Ubuntu systems, install rdma-core *devel* pkgs (e.g., rdma-core-devel, libibverbs-devel, libmlx5-devel)."
    exit 1
  fi
fi

# 3) Rust toolchain
if ! command -v cargo >/dev/null 2>&1; then
  say "Installing rustup ($RUST_TOOLCHAIN)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain "$RUST_TOOLCHAIN"
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi
rustup toolchain install "$RUST_TOOLCHAIN" -y >/dev/null 2>&1 || true
rustup default "$RUST_TOOLCHAIN" >/dev/null 2>&1 || true

# 4) Get or update repo
if [ ! -d "$MONARCH_DIR/.git" ]; then
  say "Cloning meta-pytorch/monarch -> $MONARCH_DIR"
  git clone https://github.com/meta-pytorch/monarch.git "$MONARCH_DIR"
else
  say "Pulling latest in $MONARCH_DIR"
  git -C "$MONARCH_DIR" pull --ff-only
fi

# 5) Python build tooling
say "Upgrading pip/setuptools/wheel/maturin"
"$PYO3_PYTHON" -m pip install -U pip setuptools wheel maturin

# 6) Install PyTorch Nightly (auto-detect CUDA/ROCm)
detect_accel() {
  local a="cpu"
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cv
    cv="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1 || true)"
    case "$cv" in
      12.8*) a="cu128" ;;
      12.6*) a="cu126" ;;
      *)     a="cu128" ;;
    esac
  elif [ -d /opt/rocm ] || command -v rocminfo >/dev/null 2>&1; then
    a="rocm6.3"
  fi
  printf "%s" "$a"
}
if [ "$TORCH_ACCELERATOR" = "auto" ]; then
  TORCH_ACCELERATOR="$(detect_accel)"
fi
say "Installing PyTorch NIGHTLY for: $TORCH_ACCELERATOR"
"$PYO3_PYTHON" -m pip install --pre -U \
  torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/nightly/${TORCH_ACCELERATOR}"

# 7) Put torch private libs (CUDA/ROCm/NCCL) on runtime path
TORCH_LIB_DIR="$("$PYO3_PYTHON" - <<'PY'
import pathlib
try:
    import torch
    print(pathlib.Path(torch.__file__).parent / "lib")
except Exception:
    print("")
PY
)"
if [ -n "$TORCH_LIB_DIR" ] && [ -d "$TORCH_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# 8) Clean any old cargo config / flags
say "Purging prior Cargo config and RUSTFLAGS"
rm -rf "$MONARCH_DIR/.cargo" || true
unset RUSTFLAGS || true

# 9) Use clang and apply Fix A (append -llzma at the end)
export CC=clang
export CXX=clang++

say "Configuring linker wrapper (Fix A: append -llzma)"
WRAP="/tmp/cc_lzma_wrap.sh"
cat > "$WRAP" <<'SH'
#!/usr/bin/env bash
exec "${CC:-cc}" "$@" -Wl,-Bdynamic -llzma
SH
chmod +x "$WRAP"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$WRAP"

# 10) Python reqs from repo
say "Installing Monarch python requirements"
"$PYO3_PYTHON" -m pip install -r "$MONARCH_DIR/build-requirements.txt"
"$PYO3_PYTHON" -m pip install -r "$MONARCH_DIR/python/tests/requirements.txt"

# 11) Build & install
say "Cleaning previous build artifacts"
rm -rf "$MONARCH_DIR/target"

say "Building Monarch with USE_TENSOR_ENGINE=$USE_TENSOR_ENGINE"
cd "$MONARCH_DIR"
USE_TENSOR_ENGINE="$USE_TENSOR_ENGINE" "$PYO3_PYTHON" -m pip install --no-build-isolation -v .

# 12) Smoke test
say "Verifying import"
"$PYO3_PYTHON" - <<'PY'
import sys, torch
print("python:", sys.version.split()[0])
print("torch :", torch.__version__)
import monarch
print("monarch import OK")
PY

say "Done."
