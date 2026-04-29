# Fast Image-to-3D Generation Server

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srivtx/img-to-3d/blob/main/colab/Image_to_3D_Generator.ipynb)

Progressive (coarse → fine) image-to-3D generation API + web UI.

**Current status:** Full UI/API works. Mock mode = placeholder sphere. Real 3D requires InstantMesh setup below.

**Quick options:**
- 🚀 **[Open in Colab](https://colab.research.google.com/github/srivtx/img-to-3d/blob/main/colab/Image_to_3D_Generator.ipynb)** — Free T4 GPU, run in 5 minutes
- 🐳 [Deploy to Hugging Face Spaces](HUGGINGFACE_DEPLOY.md) — Persistent URL, use your credits
- 💻 [Run locally](#quick-start-works-immediately) — Your Mac, no setup

---

## Quick Start (Works Immediately)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Run server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Open app
open http://localhost:8000
```

**Upload an image** → watch progress → see 3D sphere in viewer → download GLB.

The sphere is a **placeholder**. For real object generation, follow the InstantMesh setup below.

---

## What Works Now vs. What Needs Setup

| Feature | Status | Needs Setup? |
|---------|--------|--------------|
| Upload UI + 3D viewer | ✅ Works | No |
| Progress polling + downloads | ✅ Works | No |
| API endpoints | ✅ Works | No |
| Mock 3D (sphere) | ✅ Works | No |
| **Real 3D from photos** | ⚠️ Mock only | **Yes — InstantMesh** |
| GPU acceleration | ✅ Auto-detects | No |

---

## Setup Real 3D Generation (InstantMesh)

### Step 1: Clone & Install (One Command)

```bash
./setup_instantmesh.sh
```

Or manually:
```bash
cd models
git clone https://github.com/TencentARC/InstantMesh.git
cd InstantMesh
pip install -r requirements.txt
```

### Step 2: Download Weights (~2-4 GB)

```bash
cd models/InstantMesh

# Install huggingface CLI if needed
pip install huggingface-hub

# Download weights
huggingface-cli download TencentARC/InstantMesh --local-dir ../instantmesh
```

Or manually from: https://huggingface.co/TencentARC/InstantMesh

### Step 3: Enable Real Inference

Edit `app/pipeline/instantmesh.py`:

```python
USE_INSTANTMESH = True   # Change from False to True
```

### Step 4: Implement the Actual Forward Pass

The file has a template `_run_inference()` method. After cloning, inspect InstantMesh's actual API in `models/InstantMesh/` and replace the placeholder with their real inference code. The wrapper already handles:
- Device selection (CUDA / MPS / CPU)
- MPS fallback to CPU on unsupported ops
- FP16 only on CUDA
- Model caching in memory

### Step 5: Verify

```bash
python check_setup.py
```

Should show: `✓ READY for real inference!`

Then restart the server.

---

## Mac / Apple Silicon (M1/M2/M3)

Auto-detects MPS. If an operation fails on MPS, it automatically falls back to CPU for that layer.

```bash
# Check detected device
python -c "from app.core.config import DEVICE; print(DEVICE)"
# → mps
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate-3d` | Upload image, start generation |
| `GET` | `/jobs/{job_id}` | Poll job status & get model URLs |
| `GET` | `/health` | System status |
| `GET` | `/` | Frontend app |

### Example API Usage

```bash
# Upload
curl -X POST "http://localhost:8000/generate-3d" \
  -F "image=@your_image.jpg"

# Poll until complete
curl "http://localhost:8000/jobs/{job_id}"
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | Auto | PyTorch device (auto: cuda → mps → cpu) |
| `FP16` | `true` | Half precision (auto-disabled on CPU/MPS) |
| `KEEP_MODELS_IN_MEMORY` | `true` | Preload models on startup |
| `MAX_CONCURRENT_JOBS` | `2` | Parallel job limit |
| `REFINE_SUBDIVISIONS` | `1` | Mesh subdivision passes |

---

## Project Structure

```
image-to-3d/
├── app/
│   ├── main.py                  # FastAPI + frontend
│   ├── core/                    # Config, models
│   ├── services/                # Queue, mesh processor
│   └── pipeline/
│       ├── instantmesh.py       # ← SETUP REAL 3D HERE
│       └── refinement.py        # Mesh upsampling
├── frontend/
│   ├── index.html               # Upload UI
│   ├── style.css                # Dark theme
│   └── app.js                   # Three.js viewer
├── models/                      # InstantMesh code + weights
│   ├── InstantMesh/             # Cloned repo
│   └── instantmesh/             # Downloaded weights
├── outputs/                     # Generated GLB files
├── setup_instantmesh.sh         # One-command setup
├── check_setup.py               # Verify everything
└── test_api.py                  # API test
```

---

## Troubleshooting

### "Nothing happens when I upload"
- Check browser console for errors
- Make sure you're using `http://localhost:8000` (not `0.0.0.0`)
- Hard refresh: `Cmd + Shift + R`

### "Still showing sphere, not real 3D"
- Run `python check_setup.py` to see what's missing
- Make sure `USE_INSTANTMESH = True`
- Verify weights downloaded to `models/instantmesh/`

### "MPS error on Mac"
- Already handled — falls back to CPU automatically
- To force CPU: `export DEVICE=cpu` before running server

---

## License

MIT
