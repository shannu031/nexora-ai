#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' 'WORKER1_DIAGNOSTICS_V2 shell entry'
printf 'Worker 1 launcher shell path: %s\n' "${BASH_SOURCE[0]}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f worker1_start.py ]] || ! grep -Fq 'WORKER1_DIAGNOSTICS_V2' worker1_start.py; then
  printf '%s\n' 'Stale Worker 1 copy detected: worker1_start.py lacks WORKER1_DIAGNOSTICS_V2.' >&2
  exit 1
fi

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.13" ]]; then
  printf 'Expected Kaggle Python 3.13; found Python %s.\n' "$PYTHON_VERSION" >&2
  exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  printf '%s\n' 'Worker 1 native build currently requires Linux x86_64.' >&2
  exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq build-essential cmake ninja-build
else
  printf '%s\n' 'apt-get is required for the reproducible glibc source build.' >&2
  exit 1
fi

python -m pip uninstall -y llama-cpp-python >/dev/null 2>&1 || true
python -m pip install --disable-pip-version-check --quiet --upgrade pip setuptools wheel
CMAKE_ARGS='-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=OFF -DLLAMA_BUILD_TESTS=OFF'
CMAKE_BUILD_PARALLEL_LEVEL=2
export CMAKE_ARGS CMAKE_BUILD_PARALLEL_LEVEL
LLAMA_CPP_SDIST_URL='https://files.pythonhosted.org/packages/d7/0f/b26ceec20294f5d592ac137ccc2f10e91fa1f35bb8555906b5718a625ac1/llama_cpp_python-0.3.19.tar.gz'
LLAMA_CPP_SDIST_SHA256='6816e4a0a74ae9a356e7a0546e13147543f01f695c86a28d940e4b86519acf13'
LLAMA_CPP_SDIST_PATH="${TMPDIR:-/tmp}/llama_cpp_python-0.3.19.tar.gz"
curl --silent --show-error --fail --location --output "$LLAMA_CPP_SDIST_PATH" "$LLAMA_CPP_SDIST_URL"
printf '%s  %s\n' "$LLAMA_CPP_SDIST_SHA256" "$LLAMA_CPP_SDIST_PATH" | sha256sum --check --status || {
  rm -f "$LLAMA_CPP_SDIST_PATH"
  printf '%s\n' 'llama-cpp-python source download failed SHA-256 verification.' >&2
  exit 1
}
python -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --force-reinstall \
  --no-binary=:all: \
  "$LLAMA_CPP_SDIST_PATH"

python -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-binary=llama-cpp-python \
  -r requirements-worker1.txt

case "$(uname -m)" in
  x86_64)
    CLOUD_FLARED_ARCH=amd64
    CLOUDFLARED_SHA256=df13e7e0a027f648c410b5cc701fbcff028724d0e93209796cdbb79ec38695d4
    ;;
  aarch64|arm64)
    CLOUD_FLARED_ARCH=arm64
    CLOUDFLARED_SHA256=2561391ee9abdc828afcc52f5ac314b6551a9a6a31ff59cbc64efc63aec04615
    ;;
  *) printf '%s\n' 'Unsupported CPU architecture for cloudflared.' >&2; exit 1 ;;
esac

CLOUDFLARED_VERSION=2025.4.0
CLOUDFLARED_PATH="${TMPDIR:-/tmp}/nexora-cloudflared-${CLOUDFLARED_VERSION}"
CLOUDFLARED_URL="https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-${CLOUD_FLARED_ARCH}"

curl --silent --show-error --fail --location --output "$CLOUDFLARED_PATH" "$CLOUDFLARED_URL"
printf '%s  %s\n' "$CLOUDFLARED_SHA256" "$CLOUDFLARED_PATH" | sha256sum --check --status || {
  rm -f "$CLOUDFLARED_PATH"
  printf '%s\n' 'The cloudflared download failed checksum verification.' >&2
  exit 1
}
chmod 700 "$CLOUDFLARED_PATH"

exec python -u worker1_start.py --cloudflared "$CLOUDFLARED_PATH"