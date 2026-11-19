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

python -m pip install \
  --disable-pip-version-check \
  --quiet \
  --only-binary=llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
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