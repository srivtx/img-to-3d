"""Generate the Colab notebook from Python.

Always edit THIS file. Never edit the .ipynb JSON by hand — that's how the
escape-sequence corruption from previous handoffs happened. Run
``python generate_colab.py`` after every change to refresh
``colab/Image_to_3D_Generator.ipynb``.
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

INTRO_MD = """\
# Image-to-3D Generator on Colab (Free T4 GPU)

**What this does:**
1. Clones the repo
2. Installs everything (including InstantMesh's pinned deps)
3. Downloads InstantMesh AI weights (~4 GB)
4. Runs a pre-flight diagnostic so you know REAL vs MOCK mode before generating
5. Starts the server and a public tunnel

**Runtime:** ~5-8 minutes first time (downloads weights)
**GPU:** Free T4 (16 GB VRAM)

---

## Step 1: Check GPU

Make sure you have a GPU assigned. If not: Runtime → Change runtime type → GPU
"""

CHECK_GPU_PY = """\
!nvidia-smi
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
"""

INSTALL_MD = """\
## Step 2: Clone repo + InstantMesh + install everything

This downloads the application code, clones InstantMesh, and installs all
Python packages (including InstantMesh's pinned ``diffusers``,
``transformers``, ``pytorch-lightning`` etc, and the CUDA-dependent
``nvdiffrast``).
"""

# Note: this cell fixes the long-standing "shell-init: getcwd" bug. If the
# user re-runs this cell after a previous run, the shell's CWD is still
# /content/img-to-3d. Calling shutil.rmtree on it from there leaves the
# shell in a deleted directory and every subsequent ``!command`` fails.
# We chdir to /content FIRST.
INSTALL_PY = """\
import os
import shutil
import subprocess
import time
import importlib.util

# IMPORTANT: don't delete the directory we're standing in. Re-running this
# cell after a previous run leaves CWD = /content/img-to-3d, and rmtree on
# that puts the shell in a "ghost" directory where every shell command
# (including the very next !git) fails with "shell-init: getcwd".
os.chdir("/content")

if os.path.exists("/content/img-to-3d"):
    shutil.rmtree("/content/img-to-3d")


def git_clone_with_retry(url, dest, *, attempts=5, depth=1):
    \"\"\"
    Robust git clone. Colab/GitHub occasionally drops connections mid-clone
    with "Failed to connect to github.com port 443 after 136003 ms" -- without
    a retry loop the rest of the cell silently runs without the cloned code.
    \"\"\"
    if os.path.exists(dest) and os.listdir(dest):
        print(f"[clone] {dest} already exists, skipping")
        return
    if os.path.exists(dest):
        shutil.rmtree(dest)

    last_err = None
    for attempt in range(1, attempts + 1):
        cmd = ["git", "clone", "--depth", str(depth), url, dest]
        print(f"[clone] attempt {attempt}/{attempts}: {' '.join(cmd)}")
        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            print(f"[clone] OK -> {dest}")
            return
        last_err = f"git exit {proc.returncode}"
        wait = min(30, 5 * attempt)
        print(f"[clone] failed ({last_err}); retrying in {wait}s ...")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        time.sleep(wait)
    raise RuntimeError(f"Could not clone {url} after {attempts} attempts: {last_err}")


# 1. Application code
git_clone_with_retry("https://github.com/srivtx/img-to-3d.git", "/content/img-to-3d")
%cd /content/img-to-3d

# 2. Base Python deps for the FastAPI app
!pip install -r requirements.txt

# 3. cloudflared (public tunnel)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

# 4. InstantMesh source code (we use src/utils/* and src/models/* directly).
#    Use the same retry helper so a transient github.com timeout doesn't
#    leave us with an empty models/InstantMesh dir and a server that runs
#    in MOCK mode forever.
os.makedirs("models", exist_ok=True)
git_clone_with_retry(
    "https://github.com/TencentARC/InstantMesh.git",
    "/content/img-to-3d/models/InstantMesh",
)

# 5. InstantMesh runtime deps -- TARGETED list.
#
#    We deliberately do NOT use `pip install -r models/InstantMesh/requirements.txt`.
#    That file pins old versions (gradio==3.41.2, bitsandbytes, xformers==0.0.22.post7,
#    deepspeed, diffusers==0.19.3, transformers==4.34.1, torch==2.1.0) that don't
#    have wheels for the Python 3.12 / torch 2.5+ that current Colab ships, and pip
#    silently keeps going past the failures, leaving you with rembg / pytorch_lightning /
#    nvdiffrast / xatlas missing.
#
#    rembg is pinned to <2.0.70 because newer rembg requires numpy>=2.3 and
#    pillow>=12.1 which would force-upgrade the numpy/pillow that Colab has
#    ALREADY imported in the kernel, triggering "WARNING: you must restart the
#    runtime" and breaking the rest of this cell. Older rembg has the same API
#    we use (rembg.new_session, rembg.remove) and accepts whatever numpy/pillow
#    Colab ships with.
#
#    ninja is required for nvdiffrast's CUDA build (next step).
!pip install \\
    "pytorch-lightning>=2.0" \\
    "omegaconf" \\
    "einops" \\
    "rembg>=2.0.50,<2.0.70" \\
    "xatlas" \\
    "plyfile" \\
    "PyMCubes" \\
    "opencv-python" \\
    "imageio" \\
    "ninja" \\
    "diffusers>=0.27,<1.0" \\
    "transformers>=4.36"

# 6. nvdiffrast: builds CUDA kernels from source.
#
#    --no-build-isolation is REQUIRED. nvdiffrast's setup.py imports
#    torch.utils.cpp_extension; without this flag pip creates an isolated build
#    env that doesn't have torch and setup.py exits with the (usually invisible)
#    "ERROR! Cannot compile nvdiffrast CUDA extension. ... You run 'pip install'
#    with --no-build-isolation flag" message.
#
#    This step takes 3-5 minutes of SILENT CUDA compilation. Don't kill it.
print()
print(">>> Building nvdiffrast from source. This compiles ~20 CUDA files with")
print(">>> nvcc and is silent for 3-5 minutes. Please wait, do NOT interrupt.")
!pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast/"

# 7. Smoke-test EVERYTHING right now so we surface failures here, not later.
print()
print("Verifying install...")

required_modules = [
    "torch", "diffusers", "transformers", "rembg",
    "omegaconf", "einops", "pytorch_lightning",
    "nvdiffrast", "xatlas", "imageio", "cv2",
]
missing_modules = [m for m in required_modules if importlib.util.find_spec(m) is None]

required_paths = [
    "/content/img-to-3d/app/main.py",
    "/content/img-to-3d/models/InstantMesh/src",
    "/content/img-to-3d/models/InstantMesh/configs/instant-mesh-large.yaml",
]
missing_paths = [p for p in required_paths if not os.path.exists(p)]

if missing_modules or missing_paths:
    print()
    print("WARNING: install is INCOMPLETE.")
    if missing_modules:
        print("  Missing Python modules:")
        for m in missing_modules:
            print("    -", m)
    if missing_paths:
        print("  Missing files / repos:")
        for p in missing_paths:
            print("    -", p)
    print()
    print("Scroll up in this cell's output to find the failing line")
    print("(common: a `git clone` timeout or a `pip install` build error),")
    print("then re-run THIS cell. The retry-clone helper makes that safe.")
else:
    print("All required modules and source paths OK.")
"""

WEIGHTS_MD = """\
## Step 3: Download InstantMesh weights

Downloads the AI model (~4 GB) — the brain that turns your photos into 3D.

*Only needs to run once per session. If you re-run the install cell above,
re-run this one too.*
"""

WEIGHTS_PY = """\
import os
from huggingface_hub import snapshot_download

weights_dir = "/content/img-to-3d/models/instantmesh"
os.makedirs(weights_dir, exist_ok=True)

print("Downloading InstantMesh weights from TencentARC/InstantMesh ...")
print("(~4 GB, takes 3-5 minutes the first time)")

snapshot_download(
    repo_id="TencentARC/InstantMesh",
    local_dir=weights_dir,
    local_dir_use_symlinks=False,
)

print()
print(f"Weights at: {weights_dir}")
for f in sorted(os.listdir(weights_dir)):
    p = os.path.join(weights_dir, f)
    size_mb = os.path.getsize(p) / (1024 * 1024)
    print(f"  {f:50s} {size_mb:8.1f} MB")
"""

PREFLIGHT_MD = """\
## Step 4: Pre-flight check

Before we boot the server, verify every piece InstantMesh needs is in place.
If anything is missing the server would silently fall back to MOCK mode
(icosphere) — this cell prints the exact reason BEFORE you waste time
generating.
"""

PREFLIGHT_PY = """\
import os
import importlib.util

problems = []

# 1. InstantMesh source layout
imesh_root = "/content/img-to-3d/models/InstantMesh"
for sub in ["src", "src/utils", "configs"]:
    if not os.path.exists(os.path.join(imesh_root, sub)):
        problems.append(f"InstantMesh missing: {sub}")
if not os.path.exists(os.path.join(imesh_root, "configs/instant-mesh-large.yaml")):
    problems.append("InstantMesh config missing: configs/instant-mesh-large.yaml")

# 2. Weight files
weights_dir = "/content/img-to-3d/models/instantmesh"
required_weights = ["diffusion_pytorch_model.bin", "instant_mesh_large.ckpt"]
for f in required_weights:
    p = os.path.join(weights_dir, f)
    if not os.path.exists(p):
        problems.append(f"Weight missing: {f}")
    elif os.path.getsize(p) < 100 * 1024 * 1024:  # < 100 MB = clearly truncated
        problems.append(f"Weight looks truncated: {f} ({os.path.getsize(p)} bytes)")

# 3. Critical Python deps (the ones that have failed for us in the past)
required_modules = [
    "torch", "diffusers", "transformers", "rembg",
    "omegaconf", "einops", "pytorch_lightning",
    "nvdiffrast", "xatlas", "imageio",
]
for mod in required_modules:
    if importlib.util.find_spec(mod) is None:
        problems.append(f"Python module missing: {mod}")

if problems:
    print("PRE-FLIGHT FAILED:")
    for p in problems:
        print(" -", p)
    print()
    print("The server will run in MOCK mode (icosphere) until these are fixed.")
    print("Re-run the install cell, then re-run this cell.")
else:
    print("PRE-FLIGHT PASSED")
    print("Real InstantMesh inference should work end-to-end.")
"""

SERVER_MD = """\
## Step 5: Start the server + public tunnel

Boots the FastAPI server (with ``USE_INSTANTMESH=true``) and opens a
cloudflared tunnel. Wait for the ``https://...trycloudflare.com`` URL.

The first generation per session takes ~30 s extra while the model warms up
on the GPU. Subsequent generations are ~5-10 s.
"""

# We escape backslashes carefully so the GENERATED Python source contains
# exactly: re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
SERVER_PY = """\
import os
import re
import socket
import subprocess
import sys
import threading
import time

env = os.environ.copy()
env["DEVICE"] = "cuda"
env["USE_INSTANTMESH"] = "true"
env["FP16"] = "true"
env["KEEP_MODELS_IN_MEMORY"] = "true"
env["INSTANTMESH_CONFIG"] = "instant-mesh-large"

server_log_path = "/content/server.log"
server_log_file = open(server_log_path, "w", buffering=1)

print(f"Starting FastAPI server on :8000 (logs -> {server_log_path}) ...")
server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app",
     "--host", "0.0.0.0", "--port", "8000"],
    cwd="/content/img-to-3d",
    env=env,
    stdout=server_log_file,
    stderr=subprocess.STDOUT,
    text=True,
)


def wait_for_port(port, host="127.0.0.1", timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_proc.poll() is not None:
            return False, f"server exited with code {server_proc.returncode}"
        try:
            with socket.create_connection((host, port), timeout=1):
                return True, "ok"
        except OSError:
            time.sleep(1)
    return False, f"port {port} did not open within {timeout}s"


print("Waiting for server (this is where the 4 GB of weights load into VRAM)...")
ok, msg = wait_for_port(8000, timeout=300)
if not ok:
    print(f"Server FAILED to come up: {msg}")
    print()
    print("--- last 4 KB of server log ---")
    with open(server_log_path) as f:
        print(f.read()[-4000:])
    raise SystemExit(1)
print("Server is listening on :8000")

# --- cloudflared tunnel ---------------------------------------------------
print("Starting cloudflared tunnel...")
tunnel_proc = subprocess.Popen(
    ["./cloudflared", "tunnel", "--no-autoupdate", "--url", "http://localhost:8000"],
    cwd="/content/img-to-3d",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

public_url = None
url_re = re.compile(r"https://[a-z0-9-]+\\.trycloudflare\\.com")


def read_tunnel_output():
    global public_url
    for line in iter(tunnel_proc.stdout.readline, ""):
        if not line:
            break
        m = url_re.search(line)
        if m and not public_url:
            public_url = m.group(0)
            print()
            print("=" * 60)
            print("YOUR APP IS LIVE!")
            print("=" * 60)
            print()
            print(f"Public URL: {public_url}")
            print()
            print("Open it in your browser, drop a photo, watch a real 3D")
            print("mesh appear. Keep THIS cell running -- stopping it stops")
            print("both the server and the tunnel.")
            print("=" * 60)
            print()


threading.Thread(target=read_tunnel_output, daemon=True).start()

for _ in range(60):
    if public_url:
        break
    time.sleep(1)

if not public_url:
    print("Tunnel URL not detected after 60s -- check cloudflared output above.")

print()
print("Server PID:", server_proc.pid)
print("Tail server log with:  !tail -n 200 /content/server.log")
"""

DONE_MD = """\
## Done!

Your app is running at the URL printed above.

**Important:**
- Keep this Colab tab open
- The tunnel URL is temporary
- First generation per session is slower (model warmup)

**Performance on T4:**
- Preview (coarse): ~5-10 seconds (after warmup)
- Final (refined): ~10-20 seconds total

---

## Troubleshooting

**"PRE-FLIGHT FAILED"** → Re-run the install + weights cells, then this cell.

**"Still seeing a mock sphere"** → Run the debug cell below; the server log
will show whether it loaded REAL or MOCK mode at startup.

**"CUDA out of memory"** → Restart runtime (Runtime → Disconnect and delete
runtime) and run all cells again.

**"URL not opening"** → Wait 30 s and retry. If it still fails, the tunnel
may have been throttled — re-run the server cell to get a new URL.
"""

DEBUG_PY = """\
print("--- last 200 lines of /content/server.log ---")
try:
    with open("/content/server.log") as f:
        lines = f.readlines()
    print("".join(lines[-200:]))
except FileNotFoundError:
    print("(server log not created yet -- the server cell hasn't run)")

print()
print("--- /health ---")
import urllib.request, json
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
        print(json.dumps(json.loads(r.read()), indent=2))
except Exception as e:
    print("health check failed:", e)
"""


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------

def main():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(INTRO_MD),
        new_code_cell(CHECK_GPU_PY),
        new_markdown_cell(INSTALL_MD),
        new_code_cell(INSTALL_PY),
        new_markdown_cell(WEIGHTS_MD),
        new_code_cell(WEIGHTS_PY),
        new_markdown_cell(PREFLIGHT_MD),
        new_code_cell(PREFLIGHT_PY),
        new_markdown_cell(SERVER_MD),
        new_code_cell(SERVER_PY),
        new_markdown_cell(DONE_MD),
        new_code_cell(DEBUG_PY),
    ]

    nb.metadata = {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    }

    nbformat.validate(nb)

    out_path = "colab/Image_to_3D_Generator.ipynb"
    with open(out_path, "w") as f:
        nbformat.write(nb, f)

    print(f"Wrote {out_path}")
    print("JSON valid")


if __name__ == "__main__":
    main()
