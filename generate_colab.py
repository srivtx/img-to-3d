import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

# Cell 1: Title
cells = [
    new_markdown_cell("""# Image-to-3D Generator on Colab (Free T4 GPU)

**What this does:**
1. Clones the repo
2. Installs everything
3. Downloads InstantMesh AI weights (~4GB)
4. Starts the server
5. Gives you a public URL

**Runtime:** ~5-8 minutes first time (downloads weights)
**GPU:** Free T4 (16GB VRAM)

---

## Step 1: Check GPU

Make sure you have a GPU assigned. If not: Runtime → Change runtime type → GPU"""),

    new_code_cell("""!nvidia-smi
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")"""),

    new_markdown_cell("""## Step 2: Clone Repo & Install

This downloads the application code and installs Python packages."""),

    new_code_cell("""# Clone the repo
!git clone https://github.com/srivtx/img-to-3d.git
%cd img-to-3d

# Install dependencies
!pip install -q -r requirements.txt

# Install Hugging Face hub for weight downloads
!pip install -q huggingface-hub

# Install cloudflared for tunnel (public URL)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

print("Setup complete")"""),

    new_markdown_cell("""## Step 3: Download InstantMesh Weights

Downloads the AI model (~4GB). This is the brain that turns your photos into 3D.

*Only needs to run once per session.*"""),

    new_code_cell("""import os
from huggingface_hub import snapshot_download

weights_dir = "/content/img-to-3d/models/instantmesh"
os.makedirs(weights_dir, exist_ok=True)

print("Downloading InstantMesh weights... This takes 3-5 minutes")
print("(~4GB total)")

snapshot_download(
    repo_id="TencentARC/InstantMesh",
    local_dir=weights_dir,
    local_dir_use_symlinks=False
)

print(f"Weights downloaded to: {weights_dir}")
print(f"Files: {len(os.listdir(weights_dir))}")"""),

    new_markdown_cell("""## Step 4: Start the Server

Starts the FastAPI server with real 3D generation enabled.

**Wait for the tunnel URL to appear below (takes ~30 seconds).**"""),

    new_code_cell("""import subprocess
import time
import threading
import re

# Environment variables for the server
env = os.environ.copy()
env["DEVICE"] = "cuda"
env["USE_INSTANTMESH"] = "true"
env["FP16"] = "true"
env["KEEP_MODELS_IN_MEMORY"] = "true"

# Start the FastAPI server in background
print("Starting server on port 8000...")
server_proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd="/content/img-to-3d",
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Give server time to start
time.sleep(5)

# Start cloudflared tunnel
print("Starting public tunnel...")
tunnel_proc = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://localhost:8000"],
    cwd="/content/img-to-3d",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

public_url = None

def read_tunnel_output():
    global public_url
    for line in iter(tunnel_proc.stdout.readline, ''):
        if not line:
            break
        match = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', line)
        if match and not public_url:
            public_url = match.group(0)
            print("\\n" + "="*60)
            print("YOUR APP IS LIVE!")
            print("="*60)
            print(f"\\nPublic URL: {public_url}")
            print("\\nClick or copy this URL to your browser\\n")
            print("Upload a photo and see 3D generation!")
            print("="*60 + "\\n")

# Read tunnel output in background thread
tunnel_thread = threading.Thread(target=read_tunnel_output, daemon=True)
tunnel_thread.start()

# Wait for URL to appear
for i in range(30):
    if public_url:
        break
    time.sleep(1)

if not public_url:
    print("Tunnel URL not found yet. Check output above.")

print("\\nServer is running. Keep this cell alive!")
print("(Stopping this cell will stop the server)")"""),

    new_markdown_cell("""## Done!

Your app is running at the URL shown above.

**Important:**
- Keep this Colab tab open
- The tunnel URL is temporary
- First generation is slower (model loads into GPU)

**Performance on T4:**
- Preview (coarse): ~5-10 seconds
- Final (refined): ~10-20 seconds

---

## Troubleshooting

**\"No GPU found\"** → Runtime → Change runtime type → GPU

**\"CUDA out of memory\"** → Restart runtime and run cells again

**\"Tunnel not working\"** → Check the debug cell below"""),

    new_code_cell("""# Debug: Show recent server output
print("Recent server output:")
print("-"*50)

import select
if server_proc and server_proc.poll() is None:
    ready, _, _ = select.select([server_proc.stdout], [], [], 0.5)
    if ready:
        for _ in range(10):
            line = server_proc.stdout.readline()
            if not line:
                break
            print(line.strip())
else:
    print("Server process status:", server_proc.poll() if server_proc else "None")""")
]

nb.cells = cells
nb.metadata = {
    "colab": {
        "provenance": [],
        "gpuType": "T4"
    },
    "kernelspec": {
        "name": "python3",
        "display_name": "Python 3"
    },
    "language_info": {
        "name": "python"
    },
    "accelerator": "GPU"
}

# Validate
nbformat.validate(nb)

# Write
with open("colab/Image_to_3D_Generator.ipynb", "w") as f:
    nbformat.write(nb, f)

print("Notebook created successfully!")
