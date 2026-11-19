# Worker 1 Recovery

Worker 1 runs Qwen 2.5 7B Instruct Q4_K_M on a Kaggle CPU session and exposes
the existing authenticated `/health` and `/generate` API through a Cloudflare
Quick Tunnel. Kaggle sessions expire; this procedure makes recovery
repeatable, not permanent.

## One-Time Setup

1. Make this repository available in a writable Kaggle session directory, such
   as `/kaggle/working/nexora-ai`.
2. In the Kaggle notebook, open **Add-ons → Secrets** and attach a secret named
   `NEXORA_WORKER_1_API_KEY` to the notebook.
3. Set that Kaggle secret to the exact same value already used by the existing
   GitHub Codespaces `NEXORA_WORKER_1_API_KEY` secret. Use the secure source
   where that existing value was originally provisioned. Do not rotate or
   replace the Codespaces key as part of this setup. GitHub does not reveal a
   saved secret value; if the existing value is not available from its secure
   source, stop and resolve that before launching Worker 1.
4. Use a Kaggle runtime reporting Python 3.13 on x86_64 Linux, such as the
   provided Kaggle environment. Enable Internet in the session. The startup
   downloads pinned Python packages, Cloudflare Tunnel, and the public model
   files. It fails before installation if the Python 3.13 runtime is absent.

The launcher reads the key with Kaggle's `UserSecretsClient` on every start. It
does not generate a key, write one to disk, or print it. The same Kaggle secret
therefore remains stable across Kaggle session restarts. The Codespaces API-key
secret remains unchanged.

## Native llama-cpp Wheel

Worker 1 pins `llama-cpp-python==0.3.36` and installs the verified Python 3.13
Linux x86_64 CPU wheel from the `abetlen.github.io` CPU wheel index, with PyPI
remaining the primary index for other requirements. Installation uses
`--only-binary=llama-cpp-python` and `--no-cache-dir`, so pip cannot fall back to
a source build or reuse the previously cached musl-linked wheel. The startup no
longer installs CMake/compiler packages or builds llama-cpp-python from source.

At the start of Worker 1 execution, `worker1_start.py` imports `llama_cpp` before
downloading model shards. It reports the package location and rejects a
`libllama.so` that names musl in `readelf -d` output. No fake libc link or musl
installation is used.

Before launching, the native package can be checked in Kaggle with:

```python
import llama_cpp
from pathlib import Path
print(llama_cpp.__file__)
print(platform_libc := __import__('platform').libc_ver())
```

```sh
readelf -d /usr/local/lib/python3.13/dist-packages/llama_cpp/lib/libllama.so | grep NEEDED
ldd /usr/local/lib/python3.13/dist-packages/llama_cpp/lib/libllama.so
```

The package should show glibc through `platform.libc_ver()`, and `ldd` should
report glibc libraries, not `libc.musl-x86_64.so.1`. The exact package path may
differ, so use `llama_cpp.__file__` to locate `libllama.so` in the first check.

## Start Or Recover

From a Kaggle notebook cell, synchronize the repository checkout explicitly:

```sh
%cd /kaggle/working/nexora-ai
!git fetch origin
!git reset --hard origin/main
!git status --short --branch
!grep -n 'WORKER1_DIAGNOSTICS_V2 shell entry' start_worker1.sh
```

Use `git reset --hard` only after confirming Kaggle has no local files worth
preserving. The model cache under `/kaggle/working/nexora-worker1-model` is not
removed by this reset.

For a full recovery, run the launcher in the foreground in a Kaggle cell. Keep
that cell running; do not use `nohup` or append `&`. The launcher supervises
both the FastAPI server and Cloudflare Tunnel, so the cell remains active while
the worker is healthy and prints a specific traceback/status if either exits.

If the wheel and Python dependencies are already installed, verify startup
without reinstalling them by running this foreground Kaggle cell command:

```sh
!NEXORA_WORKER_1_SKIP_INSTALL=1 bash /kaggle/working/nexora-ai/start_worker1.sh
```

The skip flag verifies the installed `llama-cpp-python==0.3.36` version and
glibc native libraries before continuing. It does not skip model cache checks,
health verification, or tunnel startup.

The shell marker must be the first diagnostics line:

```text
WORKER1_DIAGNOSTICS_V2 shell entry
```

If this marker is absent or the line says a stale copy was detected, the Kaggle
checkout is not running the reviewed launcher. After the implementation has
been committed and pushed, synchronize `origin/main` again before restarting.

The script pins the Cloudflare Tunnel binary and verifies its SHA-256 checksum,
installs the versions in `requirements-worker1.txt`, downloads both Q4_K_M
shards from Hugging Face revision
`bb5d59e06d9551d752d08b292a50eb208b07ab1f`, loads the first split shard with
`llama-cpp-python`, starts FastAPI on loopback, and only then starts the tunnel.
Wait for the line beginning `Worker 1 tunnel URL:`. The tunnel URL is public;
the API key is not included in that line or forwarded to the `cloudflared`
process. Keep the notebook session running while the worker is in use.
By default the API listens on `127.0.0.1:7860`; set
`NEXORA_WORKER_1_PORT` only when a different local port is required. Startup
prints the selected port and confirms the authenticated `/health` check before
opening the tunnel.

To update the URL in GitHub Codespaces without changing the API-key secret,
run `bash update_worker1_url.sh` from the repository in a terminal authenticated
to the correct GitHub account with `gh`. Paste the new URL when prompted. The
script accepts only an HTTPS `trycloudflare.com` URL and sends it through
standard input to `gh secret set` for `NEXORA_WORKER_1_URL`; it does not touch
`NEXORA_WORKER_1_API_KEY` or either Worker 2 secret. After updating the secret,
restart the Codespace or its backend process so the process environment reloads.

When Kaggle expires a session, start a new one and run the same startup command.
The model is downloaded again if it is not present in the new session. A new
Quick Tunnel normally has a different URL, so update only
`NEXORA_WORKER_1_URL` using the helper. No production secret is changed by the
startup script.

## Free Hosting Notes

Free hosted notebook and web-service tiers can still sleep or spin down, so
they do not guarantee a permanently available CPU worker. Oracle Cloud's
Always Free ARM allowance is a possible VM experiment, but it is limited to 2
OCPUs and 12 GB memory after trial and availability is not guaranteed. It has
not been benchmarked with this model, so this repository does not recommend a
migration. No paid infrastructure is required or provisioned by these scripts.

## Checks

Run the local contract and launcher-helper tests with:

```sh
python -m unittest discover -s tests -p 'test_worker1.py' -v
```

The tests mock Kaggle Secrets, model download, and model inference. A real model
load and public tunnel smoke test must be done in Kaggle with Internet enabled
and the secret attached; they are not run from the Codespace.