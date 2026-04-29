# Image-to-3D Generation Server — Project Context & Handoff

## Project Overview

A full-stack application that converts a single input image into a 3D model using a progressive (coarse → refined) pipeline.

**Goal:** Upload a photo → get a 3D model in ~10-20 seconds.

**Current Status:** Backend API, frontend UI, and queue system are fully working. The only missing piece is wiring the actual AI model (InstantMesh) for real 3D generation. Currently falls back to a mock icosphere.

---

## Architecture

```
User Uploads Image
    ↓
Frontend (Three.js + drag-drop)
    ↓ POST /generate-3d
FastAPI Backend
    ↓ Background Task
Job Queue (in-memory, async)
    ↓
Coarse Generation (InstantMesh — TODO)
    ↓ Returns preview.glb immediately
Async Refinement (mesh upsampling + smoothing)
    ↓
Final Output (final.glb)
    ↓
3D Viewer loads model
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python |
| Frontend | Vanilla HTML/CSS/JS + Three.js (importmap) |
| Queue | In-memory asyncio (JobQueue class) |
| Mesh Processing | Trimesh |
| AI Model | InstantMesh (TencentARC) — NOT YET WIRED |
| Deployment | FastAPI static files (local) / Docker (HF Spaces) / Colab |

---

## File Structure

```
image-to-3d/
├── app/
│   ├── main.py                  # FastAPI app, serves frontend + API
│   ├── core/
│   │   ├── config.py            # Settings, auto-detects CUDA/MPS/CPU
│   │   └── models.py            # Pydantic request/response models
│   ├── services/
│   │   ├── queue.py             # In-memory async job queue
│   │   └── mesh_processor.py    # Trimesh refinement (subdivide, smooth, UVs)
│   └── pipeline/
│       ├── instantmesh.py       # ← THIS IS THE BROKEN FILE
│       └── refinement.py        # Async refinement pipeline
├── frontend/
│   ├── index.html               # Upload UI + 3D viewer layout
│   ├── style.css                # Dark theme
│   └── app.js                   # Three.js viewer + polling logic
├── models/                      # InstantMesh code + weights (created at runtime)
│   ├── InstantMesh/             # Cloned from GitHub
│   └── instantmesh/             # Downloaded weights from HuggingFace
├── outputs/                     # Generated GLB files per job
├── colab/
│   └── Image_to_3D_Generator.ipynb  # Colab notebook (has JSON issues)
├── Dockerfile                   # For Hugging Face Spaces
├── requirements.txt
└── README.md
```

---

## What's Working ✅

### 1. Local Development (Mac/Any machine)
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
open http://localhost:8000
```
- Full upload UI with drag-and-drop
- Progress tracking via polling
- 3D viewer with orbit controls
- Mock mode generates an icosphere (placeholder)
- Download GLB files

### 2. API Endpoints
- `POST /generate-3d` — Upload image, returns job_id
- `GET /jobs/{job_id}` — Poll status, get preview/final URLs
- `GET /health` — System status
- `GET /` — Serves frontend

### 3. Auto Device Detection
- CUDA → NVIDIA GPUs
- MPS → Apple Silicon (with CPU fallback for unsupported ops)
- CPU → Fallback everywhere

### 4. Async Pipeline
- Upload → immediate response with job_id
- Background task processes coarse → refinement
- Frontend polls every second for updates
- Preview loads first, final replaces it when ready

---

## What's Broken ❌

### The Core Problem: `app/pipeline/instantmesh.py`

This file is supposed to load and run the InstantMesh AI model. It currently:
1. Detects if InstantMesh code + weights exist
2. Tries multiple import strategies (all failing)
3. Falls back to `_mock_generate()` which creates an icosphere

**Why it's failing in Colab:**

The error chain:
```
1. Notebook clones repo: git clone https://github.com/srivtx/img-to-3d.git
2. Notebook clones InstantMesh: git clone https://github.com/TencentARC/InstantMesh.git
3. Notebook downloads weights: huggingface-cli download TencentARC/InstantMesh
4. Server starts with USE_INSTANTMESH=true
5. instantmesh.py tries to import/run InstantMesh
6. All strategies fail → falls back to mock mode
```

**Specific issues encountered:**

#### Issue 1: Import Failures
```
No module named 'src.models.mesh_fusion'
No module named 'instantmesh.pipeline'
```

**Root cause:** InstantMesh's actual API is different from what we assumed. Their repo structure doesn't match the expected module paths.

#### Issue 2: Diffusers Pipeline Loading
```
DiffusionPipeline.from_pretrained() fails
```

**Root cause:** InstantMesh is NOT a standard Diffusers pipeline. It has its own custom inference code.

#### Issue 3: Colab Working Directory
```
shell-init: error retrieving current directory: getcwd: cannot access parent directories
```

**Root cause:** The notebook deletes `/content/img-to-3d` with `shutil.rmtree()` but then the shell's current directory is still the deleted folder. All subsequent commands fail.

#### Issue 4: Notebook JSON Corruption
Multiple `SyntaxError: JSON.parse` errors when opening the `.ipynb` in Colab.

**Root cause:** Manual edits to the `.ipynb` JSON broke escape sequences (especially `
` inside Python strings).

---

## What InstantMesh Actually Is

**Repository:** https://github.com/TencentARC/InstantMesh
**Weights:** https://huggingface.co/TencentARC/InstantMesh
**Paper:** "InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models"

**What it does:**
- Input: Single RGB image
- Output: 3D mesh (obj/glb)
- Speed: ~5-10 seconds on T4 GPU
- Method: Feed-forward (single pass, no optimization loops)

**Key files in their repo:**
- `run.py` — Main inference script
- `gradio_app.py` — Gradio demo app
- `src/` — Source code (various submodules)
- Uses diffusers, transformers, but with CUSTOM pipeline code

---

## What Needs to Be Done

### Option 1: Fix the Notebook (Easiest)

The Colab notebook needs to:
1. **Fix working directory issue** — Don't delete the folder we're in. Use `os.chdir('/content')` before `shutil.rmtree()`
2. **Fix the `.ipynb` JSON** — Use `nbformat` Python library to generate the notebook, never manual edits
3. **Run InstantMesh properly** — Instead of trying to import their modules, run their `run.py` script via subprocess with the correct arguments

### Option 2: Fix `instantmesh.py` (Proper Integration)

Instead of trying to import InstantMesh as a module (which fails), do one of:

**A. Subprocess approach (most reliable):**
```python
subprocess.run([
    "python", "models/InstantMesh/run.py",
    "--input", image_path,
    "--output", output_dir,
    "--model_path", "models/instantmesh"
])
```

**B. Import their script and call functions:**
```python
sys.path.insert(0, "models/InstantMesh")
from run import main as instantmesh_main
# Or find their actual inference function
```

**C. Use their Gradio app as a reference:**
Open `models/InstantMesh/gradio_app.py` and see how they load the model and run inference. Copy that pattern.

### Option 3: Alternative Model (If InstantMesh is too hard)

Consider using a different model that's easier to integrate:
- **CRM (CVPR 2024)** — Also feed-forward, might have simpler API
- **Zero123** — Multi-view then mesh extraction
- **TripoSR** — If available via API

---

## Current Code State

### Working Files (Don't touch)
- `app/main.py` — FastAPI app, works perfectly
- `app/services/queue.py` — Job queue, works
- `app/services/mesh_processor.py` — Mesh refinement, works
- `frontend/index.html` — Upload UI, works
- `frontend/style.css` — Styles, works
- `frontend/app.js` — Three.js viewer + polling, works
- `app/core/config.py` — Auto device detection, works
- `app/core/models.py` — Pydantic models, works
- `app/pipeline/refinement.py` — Refinement pipeline, works

### Broken Files (Fix these)
- `colab/Image_to_3D_Generator.ipynb` — JSON corruption + working directory bug
- `app/pipeline/instantmesh.py` — Can't load/run InstantMesh
- `generate_colab.py` — Script that generates the notebook

---

## Testing Locally (Without Colab)

You can test the integration locally without Colab:

```bash
# 1. Clone InstantMesh
cd models
git clone https://github.com/TencentARC/InstantMesh.git

# 2. Download weights
huggingface-cli download TencentARC/InstantMesh --local-dir instantmesh

# 3. Set env var
export USE_INSTANTMESH=true

# 4. Run server
python -m uvicorn app.main:app --reload

# 5. Check logs — it will show whether InstantMesh loaded or fell back to mock
```

---

## Error Log Summary

### Local (Mac)
- ✅ Server starts
- ✅ Frontend loads
- ✅ Upload works
- ✅ Mock sphere displays in 3D viewer
- ⚠️ `instantmesh.py` — `No module named 'src.models.mesh_fusion'`

### Colab (T4 GPU)
- ✅ GPU detected
- ❌ Working directory error after `shutil.rmtree()`
- ❌ `cloudflared` not found (because commands ran in deleted directory)
- ❌ Notebook JSON corruption (previous manual edits)
- ⚠️ Weights download successfully
- ⚠️ InstantMesh code clones successfully
- ❌ Can't open tunnel because cloudflared failed to download

---

## Recommended Next Steps

1. **Fix the Colab notebook JSON** using `nbformat` library in Python (never manual edits)
2. **Fix the working directory bug** — `os.chdir('/content')` before deleting
3. **Inspect InstantMesh's actual repo structure** — Look at `run.py` arguments
4. **Implement subprocess-based inference** in `instantmesh.py`
5. **Test locally first** (faster iteration than Colab)
6. **Deploy to Hugging Face Spaces** once working (persistent URL)

---

## GitHub Repo

```
https://github.com/srivtx/img-to-3d
```

All code is pushed there. The Colab notebook should be opened via:
```
https://colab.research.google.com/github/srivtx/img-to-3d/blob/main/colab/Image_to_3D_Generator.ipynb
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | Auto (cuda/mps/cpu) | PyTorch device |
| `USE_INSTANTMESH` | `false` | Enable real AI inference |
| `FP16` | `true` (cuda only) | Half precision |
| `KEEP_MODELS_IN_MEMORY` | `true` | Preload models |
| `MAX_CONCURRENT_JOBS` | `2` | Parallel jobs |

---

## Key Insight

**The problem is NOT the AI model or the GPU.** The model (InstantMesh) works. The GPU (T4) is sufficient. The problem is **integration** — our code doesn't know how to call InstantMesh's inference function correctly.

**Solution approach:** Run InstantMesh's `run.py` as a subprocess with the image path and output directory, then load the resulting mesh file. This is more reliable than trying to import their modules.

---

*Created: 2026-04-29*
*Author: Previous AI assistant*
*Status: Handoff to Claude for debugging and completion*
