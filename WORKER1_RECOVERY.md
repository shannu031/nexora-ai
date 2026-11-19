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
4. Enable Internet for the Kaggle session. The startup downloads pinned Python
   packages, Cloudflare Tunnel, and the public model files.

The launcher reads the key with Kaggle's `UserSecretsClient` on every start. It
does not generate a key, write one to disk, or print it. The same Kaggle secret
therefore remains stable across Kaggle session restarts. The Codespaces API-key
secret remains unchanged.

## Start Or Recover

From a Kaggle notebook cell, run:

```sh
!bash /kaggle/working/nexora-ai/start_worker1.sh
```

The script pins the Cloudflare Tunnel binary and verifies its SHA-256 checksum,
installs the versions in `requirements-worker1.txt`, downloads both Q4_K_M
shards from Hugging Face revision
`bb5d59e06d9551d752d08b292a50eb208b07ab1f`, loads the first split shard with
`llama-cpp-python`, starts FastAPI on loopback, and only then starts the tunnel.
Wait for the line beginning `Worker 1 tunnel URL:`. The tunnel URL is public;
the API key is not included in that line or forwarded to the `cloudflared`
process. Keep the notebook session running while the worker is in use.

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