import argparse
import importlib.metadata
import platform
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
MODEL_VARIANTS = {
    "1.5b": {
        "repository": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "revision": "91cad51170dc346986eccefdc2dd33a9da36ead9",
        "files": ("qwen2.5-1.5b-instruct-q4_k_m.gguf",),
        "description": "Qwen 2.5 1.5B Instruct Q4_K_M",
    },
    "7b": {
        "repository": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "revision": "bb5d59e06d9551d752d08b292a50eb208b07ab1f",
        "files": (
            "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
        ),
        "description": "Qwen 2.5 7B Instruct Q4_K_M",
    },
}
API_KEY_SECRET_NAME = "NEXORA_WORKER_1_API_KEY"
DEFAULT_MODEL_DIRECTORY = Path("/kaggle/working/nexora-worker1-model")
TUNNEL_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.IGNORECASE)


def get_model_variant():
    model_size = os.environ.get("NEXORA_WORKER_1_MODEL", "1.5b").strip().lower()
    try:
        return model_size, MODEL_VARIANTS[model_size]
    except KeyError:
        raise RuntimeError(
            "NEXORA_WORKER_1_MODEL must be either '1.5b' or '7b'."
        ) from None


_, MODEL_CONFIG = get_model_variant()
MODEL_REPOSITORY = MODEL_CONFIG["repository"]
MODEL_REVISION = MODEL_CONFIG["revision"]
MODEL_FILES = MODEL_CONFIG["files"]


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
        raise RuntimeError("The pinned Qwen GGUF model files are incomplete.")
    return snapshot_path / MODEL_FILES[0]


def validate_native_llama_cpp():
    try:
        import llama_cpp
    except Exception as error:
        raise RuntimeError(
            "llama_cpp native import failed. The Worker 1 package must use the "
            "glibc/x86_64 CPU build; musl-linked libraries are incompatible."
        ) from error

    try:
        package_version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        package_version = getattr(llama_cpp, "__version__", "unknown")

    package_dir = Path(llama_cpp.__file__).resolve().parent
    native_libraries = sorted(package_dir.glob("lib/*.so"))
    libllama = next((path for path in native_libraries if path.name == "libllama.so"), None)
    if libllama is None:
        raise RuntimeError("llama_cpp package does not contain libllama.so.")

    libc_name = platform.libc_ver()[0]
    if libc_name != "glibc":
        raise RuntimeError(
            f"Kaggle must use glibc; detected libc implementation: {libc_name or 'unknown'}"
        )

    try:
        dependency_check = subprocess.run(
            ["readelf", "-d", str(libllama)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("readelf is required to validate the native llama library.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("readelf could not inspect libllama.so.") from error

    needed = dependency_check.stdout.lower()
    if "musl" in needed:
        raise RuntimeError("libllama.so is linked to musl libc and is incompatible with Kaggle glibc.")

    print(
        f"llama-cpp-python {package_version}; glibc native import OK; "
        f"libllama.so: {libllama}",
        flush=True,
    )
    return libllama


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

    model_file_sizes = {}
    for filename in MODEL_FILES:
        model_file = Path(model_directory) / filename
        if not model_file.is_file():
            raise RuntimeError(f"Required GGUF model file is missing: {model_file}")
        model_file_sizes[filename] = model_file.stat().st_size

    print(
        f"llama-cpp-python {package_version}; CPU backend: {cpu_backend}",
        flush=True,
    )
    print(
        "Qwen GGUF model files: "
        + ", ".join(
            f"{name}={size} bytes" for name, size in model_file_sizes.items()
        ),
        flush=True,
    )
    return cpu_backend


def load_model(model_path):
    from llama_cpp import Llama

    model_path = Path(model_path)
    if not model_path.is_file():
        raise RuntimeError(f"The GGUF model file is missing: {model_path}")

    missing_model_files = [
        model_path.parent / filename
        for filename in MODEL_FILES
        if not (model_path.parent / filename).is_file()
    ]
    if missing_model_files:
        raise RuntimeError(
            "Pinned Qwen GGUF model files are required; missing: "
            + ", ".join(filename.name for filename in missing_model_files)
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


def wait_for_health(api_key, port, timeout=120, server_thread=None, server_failures=None):
    deadline = time.monotonic() + timeout
    request = Request(
        f"http://127.0.0.1:{port}/health",
        headers={"X-API-Key": api_key},
    )
    while time.monotonic() < deadline:
        if server_thread is not None and not server_thread.is_alive():
            _raise_server_stopped(server_failures, "before health check succeeded")
        try:
            with urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.5)
    raise RuntimeError("Worker 1 did not become healthy before the startup timeout.")


def _run_uvicorn(server, server_failures):
    try:
        server.run()
    except BaseException as error:
        server_failures.put(error)


def _raise_server_stopped(server_failures, phase):
    try:
        error = server_failures.get_nowait()
    except queue.Empty:
        raise RuntimeError(f"FastAPI server exited unexpectedly {phase}.") from None
    raise RuntimeError(f"FastAPI server crashed {phase}.") from error


def monitor_worker_processes(server_thread, tunnel, server_failures, poll_interval=1):
    while True:
        if not server_thread.is_alive():
            _raise_server_stopped(server_failures, "after startup")
        tunnel_exit_code = tunnel.poll()
        if tunnel_exit_code is not None:
            raise RuntimeError(
                f"Cloudflare Tunnel exited unexpectedly with status {tunnel_exit_code}."
            )
        time.sleep(poll_interval)


def _read_tunnel_output(process, output_queue):
    for line in process.stdout:
        print(f"[cloudflared] {line.rstrip()}", flush=True)
        tunnel_url = extract_tunnel_url(line)
        if tunnel_url:
            output_queue.put(tunnel_url)


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
            try:
                tunnel_url = output_queue.get(timeout=0.5)
            except queue.Empty:
                tunnel_url = None
            if tunnel_url is not None:
                print(f"Worker 1 tunnel URL: {tunnel_url}", flush=True)
                return process
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Cloudflare Tunnel exited before providing a URL (status {exit_code})."
                )
        raise RuntimeError("Cloudflare Tunnel did not provide a URL before timeout.")
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        raise


def run(cloudflared_path):
    validate_native_llama_cpp()
    api_key = read_api_key_from_kaggle()
    os.environ[API_KEY_SECRET_NAME] = api_key
    port = int(os.environ.get("NEXORA_WORKER_1_PORT", "7860"))

    print(
        f"Downloading the pinned {MODEL_CONFIG['description']} model.",
        flush=True,
    )
    model_path = download_model()
    model_directory = model_path.parent
    print(
        "Loading the CPU model file: "
        f"{model_directory / MODEL_FILES[0]}.",
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
    server_failures = queue.Queue()
    server_thread = threading.Thread(
        target=_run_uvicorn,
        args=(server, server_failures),
        daemon=True,
    )
    server_thread.start()
    tunnel = None
    try:
        print(f"Starting Worker 1 API on http://127.0.0.1:{port}", flush=True)
        wait_for_health(
            api_key,
            port,
            server_thread=server_thread,
            server_failures=server_failures,
        )
        print(f"Worker 1 health check passed: http://127.0.0.1:{port}/health", flush=True)
        tunnel = start_tunnel(cloudflared_path, port)
        monitor_worker_processes(server_thread, tunnel, server_failures)
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