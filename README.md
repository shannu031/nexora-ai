# nexora-ai
An independent AI platform powered by self-hosted open-weight models, featuring intelligent conversations, long-term memory, image understanding, and secure file storage.

Users can ask questions about uploaded text-based PDFs, DOCX, TXT, CSV, and XLSX files. Scanned PDFs are not supported because OCR is not enabled.

## Conversation persistence

Chat messages are stored in `conversations.sqlite3` beside `main.py`. The
frontend saves the active conversation ID in browser storage and reloads its
latest 20 messages when the page is opened again. New Chat starts a separate
conversation; existing database records are not removed.

The sidebar lists saved conversations by recent activity, using each
conversation's first user message as its title. Selecting one restores its
messages and makes its ready document uploads available for follow-up questions.

## Accounts and private data

Users can register with an email and a password of at least 12 characters.
Passwords are stored as salted scrypt hashes. Login issues a random, revocable
seven-day session in an HttpOnly, Secure, SameSite=Lax cookie. Logout revokes
the server-side session. Authenticated API requests are scoped to the owning
user; conversation, chat, upload, document, and file access checks do not trust
client-supplied IDs.

Set `NEXORA_ALLOWED_ORIGINS` to a comma-separated list of exact frontend
origins. It defaults to `http://localhost:8000,http://127.0.0.1:8000`. For
GitHub Codespaces, include the exact HTTPS forwarded origin for port 8000.
`NEXORA_COOKIE_SECURE` defaults to `true`; only set it to `false` for local
plain-HTTP development.

Existing records have no verified user owner. They are preserved and remain
unowned and inaccessible to accounts until a verified ownership migration is
performed; the application does not assign them to the first user to register.

Run the automated checks with `env -u NEXORA_WORKER_1_URL -u NEXORA_WORKER_1_API_KEY -u NEXORA_WORKER_2_URL -u NEXORA_WORKER_2_API_KEY python -m unittest discover -s tests -v`.

Authentication test results are documented in [TESTING.md](./TESTING.md).
Manual Android Chrome results and automated API results are listed separately.

## CPU inference workers

The backend can optionally use up to two authenticated CPU inference workers.
Configure these environment variables in GitHub Codespaces Secrets:

- `NEXORA_WORKER_1_URL`
- `NEXORA_WORKER_1_API_KEY`
- `NEXORA_WORKER_2_URL`
- `NEXORA_WORKER_2_API_KEY`

In GitHub, open **Settings → Secrets and variables → Codespaces**, add each
secret with the exact name above, and make it available to this Codespace.
Never put API keys in source files, frontend code, chat, or logs. Restart the
backend after configuring or rotating the secrets; the backend reads worker
configuration from its process environment at startup.

For each complete HTTPS URL/key pair, NEXORA checks `/health` using the
`X-API-Key` header and distributes requests round-robin among workers reporting
healthy. If a selected worker fails, NEXORA tries another healthy worker
sequentially; it never fans a request out to multiple workers simultaneously.
Worker calls have a 3-second health timeout and a 180-second generation
timeout. If no worker is configured, healthy, or able to generate a valid
reply, NEXORA falls back to the existing local Ollama model (`qwen2.5:1.5b`).
No worker API keys or URLs are embedded in the frontend, and no new public
health endpoint is added to NEXORA.

Worker 1 recovery after Kaggle session expiry is documented in
[WORKER1_RECOVERY.md](./WORKER1_RECOVERY.md). The startup uses a stable Kaggle
secret for API authentication; after a new session starts, normally only
`NEXORA_WORKER_1_URL` needs updating. Worker 2 configuration is independent.
