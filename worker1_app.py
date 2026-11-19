import hmac
import os
from threading import Lock

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    message: str


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    expected_api_key = os.environ.get("NEXORA_WORKER_1_API_KEY", "")
    if not expected_api_key:
        raise HTTPException(status_code=503, detail="Worker API key is not configured.")
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key.")


def create_app(model=None):
    worker_app = FastAPI(title="NEXORA Worker 1")
    worker_app.state.model = model
    worker_app.state.model_lock = Lock()

    @worker_app.get("/health", dependencies=[Depends(require_api_key)])
    def health():
        return {"status": "ok"}

    @worker_app.post("/generate", dependencies=[Depends(require_api_key)])
    def generate(data: GenerateRequest):
        model = worker_app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Worker model is not loaded.")
        with worker_app.state.model_lock:
            result = model.create_chat_completion(
                messages=[{"role": "user", "content": data.message}],
                max_tokens=1024,
                temperature=0.4,
                repeat_penalty=1.2,
            )
        reply = result["choices"][0]["message"]["content"]
        if not isinstance(reply, str) or not reply.strip():
            raise HTTPException(status_code=502, detail="Model returned an empty reply.")
        return {"reply": reply.strip()}

    return worker_app


app = create_app()