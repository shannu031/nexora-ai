import os
import secrets
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import worker1_app
import worker1_start


class FakeModel:
    def __init__(self):
        self.messages = None

    def create_chat_completion(self, **kwargs):
        self.messages = kwargs["messages"]
        return {"choices": [{"message": {"content": "Worker reply"}}]}


class WorkerOneApiTests(unittest.TestCase):
    def setUp(self):
        self.api_key = secrets.token_urlsafe(32)
        self.model = FakeModel()
        self.app = worker1_app.create_app(self.model)
        self.client = TestClient(self.app)
        self.api_key_patch = patch.dict(
            os.environ,
            {"NEXORA_WORKER_1_API_KEY": self.api_key},
        )
        self.api_key_patch.start()

    def tearDown(self):
        self.api_key_patch.stop()
        self.client.close()

    def test_health_requires_the_worker_api_key(self):
        self.assertEqual(self.client.get("/health").status_code, 401)
        response = self.client.get(
            "/health",
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_generate_preserves_the_worker_contract(self):
        response = self.client.post(
            "/generate",
            headers={"X-API-Key": self.api_key},
            json={"message": "NEXORA prompt"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"reply": "Worker reply"})
        self.assertEqual(
            self.model.messages,
            [{"role": "user", "content": "NEXORA prompt"}],
        )

    def test_missing_configured_key_fails_closed(self):
        with patch.dict(os.environ, {"NEXORA_WORKER_1_API_KEY": ""}):
            response = self.client.get(
                "/health",
                headers={"X-API-Key": self.api_key},
            )
        self.assertEqual(response.status_code, 503)


class WorkerOneStartupTests(unittest.TestCase):
    def test_api_key_is_read_from_kaggle_secrets_each_start(self):
        stable_key = secrets.token_urlsafe(32)

        class FakeSecretsClient:
            def get_secret(self, name):
                self.requested_name = name
                return stable_key

        fake_kaggle_secrets = types.ModuleType("kaggle_secrets")
        fake_kaggle_secrets.UserSecretsClient = FakeSecretsClient
        with patch.dict(sys.modules, {"kaggle_secrets": fake_kaggle_secrets}):
            first_key = worker1_start.read_api_key_from_kaggle()
            second_key = worker1_start.read_api_key_from_kaggle()

        self.assertEqual(first_key, stable_key)
        self.assertEqual(second_key, stable_key)

    def test_model_download_uses_pinned_repository_and_both_q4_k_m_shards(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_directory = Path(temporary_directory)

            def fake_snapshot_download(**kwargs):
                for filename in kwargs["allow_patterns"]:
                    (model_directory / filename).touch()
                self.assertEqual(kwargs["repo_id"], worker1_start.MODEL_REPOSITORY)
                self.assertEqual(kwargs["revision"], worker1_start.MODEL_REVISION)
                self.assertEqual(list(kwargs["allow_patterns"]), list(worker1_start.MODEL_FILES))
                return str(model_directory)

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.snapshot_download = fake_snapshot_download
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                model_path = worker1_start.download_model(model_directory)

        self.assertEqual(model_path.name, worker1_start.MODEL_FILES[0])

    def test_tunnel_url_detection_and_key_environment_isolation(self):
        api_key = secrets.token_urlsafe(32)
        with patch.dict(os.environ, {worker1_start.API_KEY_SECRET_NAME: api_key}):
            tunnel_environment = worker1_start.cloudflared_environment()

        self.assertNotIn(worker1_start.API_KEY_SECRET_NAME, tunnel_environment)
        self.assertIsNone(worker1_start.extract_tunnel_url("starting tunnel"))
        self.assertEqual(
            worker1_start.extract_tunnel_url(
                "Tunnel ready at https://worker-recovery.trycloudflare.com"
            ),
            "https://worker-recovery.trycloudflare.com",
        )


if __name__ == "__main__":
    unittest.main()