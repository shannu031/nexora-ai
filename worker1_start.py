import argparse
import importlib.metadata
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import uvicorn

import worker1_app


WORKER1_DIAGNOSTICS_V2 = "WORKER1_DIAGNOSTICS_V2"
MODEL_REPOSITORY = "Qwen/Qwen2.5-7B-Instruct-GGUF"
MODEL_REVISION = "bb5d59e06d9551d752d08b292a50eb208b07ab1f"
MODEL_FILES = (
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
)
API_KEY_SECRET_NAME = "NEXORA_WORKER_1_API_KEY"
DEFAULT_MODEL_DIRECTORY = Path("/kaggle/working/nexora-worker1-model")
TUNNEL_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.IGNORECASE)


def read_api_key_from_kaggle():
    try:
        from kaggle_secrets import UserSecretsClient

        api_key = UserSecretsClient().get_secret(API_KEY_SECRET_NAME)
    except Exception:
        raise RuntimeError(
            "Attach the Kaggle secret NEXORA_WORKER_1_API_KEY before starting Worker 1."
        ) from None
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError(
            "The Kaggle secret NEXORA_WORKER_1_API_KEY must contain a value."
        )
    return api_key.strip()


def download_model(model_directory=DEFAULT_MODEL_DIRECTORY):
    from huggingface_hub import snapshot_download

    model_directory = Path(model_directory)
    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            allow_patterns=list(MODEL_FILES),
            local_dir=str(model_directory),
        )
    )
    if any(not (snapshot_path / filename).is_file() for filename in MODEL_FILES):
        raise RuntimeError("The pinned Qwen GGUF model shards are incomplete.")
    return snapshot_path / MODEL_FILES[0]


def verify_llama_cpp_runtime(model_directory):
    import llama_cpp

    try:
        package_version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        package_version = getattr(llama_cpp, "__version__", "unknown")

    package_dir = Path(llama_cpp.__file__).resolve().parent
    cpu_backend = package_dir / "lib" / "libggml-cpu.so"
    if not cpu_backend.is_file():
        raise RuntimeError(
            f"llama-cpp-python {package_version} "
            "does not contain libggml-cpu.so."
        )

    shard_sizes = {}
    for filename in MODEL_FILES:
        shard_path = Path(model_directory) / filename
        if not shard_path.is_file():
            raise RuntimeError(f"Required GGUF shard is missing: {shard_path}")
        shard_sizes[filename] = shard_path.stat().st_size

    print(
        f"llama-cpp-python {package_version}; CPU backend: {cpu_backend}",
        flush=True,
    )
    print(
        "Qwen GGUF shards: "
        + ", ".join(f"{name}={size} bytes" for name, size in shard_sizes.items()),
        flush=True,
    )
    return cpu_backend


def load_model(model_path):
    from llama_cpp import Llama

    model_path = Path(model_path)
    if not model_path.is_file():
        raise RuntimeError(f"The split GGUF entry shard is missing: {model_path}")

    missing_shards = [
        model_path.parent / filename
        for filename in MODEL_FILES
        if not (model_path.parent / filename).is_file()
    ]
    if missing_shards:
        raise RuntimeError(
            "Both pinned Qwen GGUF shards are required; missing: "
            + ", ".join(filename.name for filename in missing_shards)
        )

    verify_llama_cpp_runtime(model_path.parent)

    available_ram_bytes = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_ram_bytes = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    if available_ram_bytes is not None:
        print(
            f"Available RAM before model load: {available_ram_bytes / (1024**3):.1f} GiB",
            flush=True,
        )

    cpu_count = os.cpu_count() or 1
    print(
        "CPU model configuration: "
        f"n_ctx=2048, n_batch=512, n_threads={min(cpu_count, 4)}, "
        "n_gpu_layers=0, use_mmap=True, use_mlock=False",
        flush=True,
    )
    return Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_batch=512,
        n_threads=min(cpu_count, 4),
        n_gpu_layers=0,
        use_mmap=True,
        use_mlock=False,
        verbose=True,
    )


def extract_tunnel_url(line):
    match = TUNNEL_URL_PATTERN.search(line)
    return match.group(0) if match else None


def cloudflared_environment():
    environment = os.environ.copy()
    environment.pop(API_KEY_SECRET_NAME, None)
    return environment


def wait_for_health(api_key, port, timeout=120):
    deadline = time.monotonic() + timeout
    request = Request(
        f"http://127.0.0.1:{port}/health",
        headers={"X-API-Key": api_key},
    )
    while time.monotonic() < deadline:
        try:
            with urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.5)
    raise RuntimeError("Worker 1 did not become healthy before the startup timeout.")


def _read_tunnel_output(process, output_queue):
    for line in process.stdout:
        output_queue.put(line)


def start_tunnel(cloudflared_path, port):
    process = subprocess.Popen(
        [
            cloudflared_path,
            "tunnel",
            "--no-autoupdate",
            "--url",
            f"http://127.0.0.1:{port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=cloudflared_environment(),
    )
    output_queue = queue.Queue()
    reader = threading.Thread(
        target=_read_tunnel_output,
        args=(process, output_queue),
        daemon=True,
    )
    reader.start()
    deadline = time.monotonic() + 120
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Cloudflare Tunnel exited before providing a URL.")
            try:
                line = output_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            tunnel_url = extract_tunnel_url(line)
            if tunnel_url:
                print(f"Worker 1 tunnel URL: {tunnel_url}", flush=True)
                return process
        raise RuntimeError("Cloudflare Tunnel did not provide a URL before timeout.")
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        raise


def run(cloudflared_path):
    api_key = read_api_key_from_kaggle()
    os.environ[API_KEY_SECRET_NAME] = api_key
    port = int(os.environ.get("NEXORA_WORKER_1_PORT", "7860"))

    print("Downloading the pinned Qwen 2.5 7B Instruct Q4_K_M model.", flush=True)
    model_path = download_model()
    model_directory = model_path.parent
    print(
        "Loading the CPU model with split shards: "
        f"{model_directory / MODEL_FILES[0]}, {model_directory / MODEL_FILES[1]}.",
        flush=True,
    )
    worker1_app.app.state.model = load_model(model_path)

    server = uvicorn.Server(
        uvicorn.Config(
            worker1_app.app,
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="warning",
        )
    )
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    tunnel = None
    try:
        wait_for_health(api_key, port)
        tunnel = start_tunnel(cloudflared_path, port)
        while tunnel.poll() is None and server_thread.is_alive():
            time.sleep(1)
        raise RuntimeError("Worker 1 server or Cloudflare Tunnel stopped.")
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
            tunnel.wait(timeout=10)


def main(argv=None):
    print(WORKER1_DIAGNOSTICS_V2, flush=True)
    print(f"Worker 1 launcher path: {Path(__file__).resolve()}", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloudflared", required=True)
    arguments = parser.parse_args(argv)
    try:
        run(arguments.cloudflared)
    except KeyboardInterrupt:
        print("Worker 1 stopped.", flush=True)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print(
            "Worker 1 startup failed. Complete traceback shown above.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())