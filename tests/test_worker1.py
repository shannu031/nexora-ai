import os
import secrets
import sys
import tempfile
import types
import unittest
import io
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

    def test_llama_cpp_runtime_preflight_requires_cpu_backend_and_both_shards(self):
        fake_llama_cpp = types.ModuleType("llama_cpp")
        fake_llama_cpp.__version__ = "0.3.36"

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_directory = Path(temporary_directory)
            for filename in worker1_start.MODEL_FILES:
                (model_directory / filename).touch()

            package_directory = model_directory / "llama_cpp"
            cpu_backend = package_directory / "lib" / "libggml-cpu.so"
            cpu_backend.parent.mkdir(parents=True)
            cpu_backend.touch()
            fake_llama_cpp.__file__ = str(package_directory / "__init__.py")

            with patch.dict(sys.modules, {"llama_cpp": fake_llama_cpp}), patch(
                "sys.stdout",
                new_callable=io.StringIO,
            ):
                detected_backend = worker1_start.verify_llama_cpp_runtime(model_directory)

        self.assertEqual(detected_backend, cpu_backend)

    def test_native_import_failure_names_glibc_requirement(self):
        fake_import_error = types.ModuleType("llama_cpp")

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with self.assertRaisesRegex(RuntimeError, "glibc/x86_64 CPU build"):
                worker1_start.validate_native_llama_cpp()

    def test_musl_native_dependency_is_rejected(self):
        fake_llama_cpp = types.ModuleType("llama_cpp")
        fake_llama_cpp.__version__ = "0.3.36"

        with tempfile.TemporaryDirectory() as temporary_directory:
            package_directory = Path(temporary_directory) / "llama_cpp"
            native_directory = package_directory / "lib"
            native_directory.mkdir(parents=True)
            libllama = native_directory / "libllama.so"
            libllama.touch()
            fake_llama_cpp.__file__ = str(package_directory / "__init__.py")

            fake_result = subprocess_result = types.SimpleNamespace(
                stdout="0x00000001 (NEEDED) Shared library: [libc.musl-x86_64.so.1]\n",
                stderr="",
            )
            with patch.dict(sys.modules, {"llama_cpp": fake_llama_cpp}), patch.object(
                worker1_start.importlib.metadata,
                "version",
                return_value="0.3.36",
            ), patch.object(
                worker1_start.platform,
                "libc_ver",
                return_value=("glibc", "2.39"),
            ), patch.object(
                worker1_start.subprocess,
                "run",
                return_value=fake_result,
            ):
                with self.assertRaisesRegex(RuntimeError, "musl libc"):
                    worker1_start.validate_native_llama_cpp()

    def test_model_load_exception_is_preserved(self):
        failure = RuntimeError("GGUF load failed in llama-cpp test")

        class FakeLlama:
            def __init__(self, **kwargs):
                raise failure

        fake_llama_cpp = types.ModuleType("llama_cpp")
        fake_llama_cpp.__file__ = "/tmp/llama_cpp/__init__.py"
        fake_llama_cpp.Llama = FakeLlama

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_directory = Path(temporary_directory)
            for filename in worker1_start.MODEL_FILES:
                (model_directory / filename).touch()

            package_directory = model_directory / "llama_cpp"
            cpu_backend = package_directory / "lib" / "libggml-cpu.so"
            cpu_backend.parent.mkdir(parents=True)
            cpu_backend.touch()
            fake_llama_cpp.__file__ = str(package_directory / "__init__.py")

            with patch.dict(sys.modules, {"llama_cpp": fake_llama_cpp}), patch.dict(
                os.environ,
                {"KAGGLE_RUNNING_IN_JUPYTERLAB": "1"},
            ):
                with self.assertRaisesRegex(RuntimeError, "GGUF load failed"):
                    worker1_start.load_model(model_directory / worker1_start.MODEL_FILES[0])

    def test_download_urls_are_plain_and_free_of_markdown(self):
        shell_text = Path(__file__).resolve().parents[1].joinpath("start_worker1.sh").read_text()
        cloudflared_url = (
            'CLOUDFLARED_URL="https://github.com/cloudflare/cloudflared/releases/download/'
            '${CLOUDFLARED_VERSION}/cloudflared-linux-${CLOUD_FLARED_ARCH}"'
        )
        self.assertIn(cloudflared_url, shell_text)
        self.assertNotIn("LLAMA_CPP_SDIST_URL", shell_text)
        self.assertNotIn("](", shell_text)
        self.assertNotIn("[https://", shell_text)

    def test_shell_entry_uses_pinned_cpu_wheel_without_source_build(self):
        shell_text = Path(__file__).resolve().parents[1].joinpath("start_worker1.sh").read_text()
        requirements_text = Path(__file__).resolve().parents[1].joinpath("requirements-worker1.txt").read_text()
        self.assertIn("llama-cpp-python==0.3.36", requirements_text)
        self.assertIn("--index-url https://pypi.org/simple", shell_text)
        self.assertIn("--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu", shell_text)
        self.assertIn("--only-binary=llama-cpp-python", shell_text)
        self.assertIn("--no-cache-dir", shell_text)
        self.assertNotIn("LLAMA_CPP_SDIST_URL", shell_text)
        self.assertNotIn("--no-binary", shell_text)
        self.assertNotIn("CMAKE_ARGS", shell_text)
        self.assertNotIn("apt-get install", shell_text)

    def test_shell_entry_prints_marker_and_runs_python_unbuffered(self):
        shell_text = Path(__file__).resolve().parents[1].joinpath("start_worker1.sh").read_text()
        self.assertIn("WORKER1_DIAGNOSTICS_V2 shell entry", shell_text)
        self.assertIn("exec python -u worker1_start.py", shell_text)
        self.assertIn("Stale Worker 1 copy detected", shell_text)
        self.assertIn("WORKER1_DIAGNOSTICS_V2", shell_text)

    def test_main_prints_version_marker_before_complete_traceback(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(worker1_start, "run", side_effect=RuntimeError("exact load error")), patch.object(
            sys,
            "argv",
            ["worker1_start.py", "--cloudflared", "/tmp/cloudflared"],
        ), patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            exit_code = worker1_start.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("WORKER1_DIAGNOSTICS_V2", stdout.getvalue())
        traceback_output = stderr.getvalue()
        traceback_position = traceback_output.find("Traceback (most recent call last)")
        exception_position = traceback_output.find("RuntimeError: exact load error")
        generic_position = traceback_output.find("Worker 1 startup failed")
        self.assertGreaterEqual(traceback_position, 0)
        self.assertGreater(exception_position, traceback_position)
        self.assertGreater(generic_position, exception_position)

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