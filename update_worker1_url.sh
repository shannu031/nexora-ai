#!/usr/bin/env bash
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  printf '%s\n' 'GitHub CLI (gh) is required.' >&2
  exit 1
fi

read -r -p 'Paste the new Worker 1 HTTPS tunnel URL: ' WORKER_1_URL
if [[ ! "$WORKER_1_URL" =~ ^https://[a-z0-9-]+\.trycloudflare\.com$ ]]; then
  unset WORKER_1_URL
  printf '%s\n' 'Expected a quick-tunnel URL ending in .trycloudflare.com.' >&2
  exit 1
fi

printf '%s' "$WORKER_1_URL" | gh secret set NEXORA_WORKER_1_URL --app codespaces
unset WORKER_1_URL