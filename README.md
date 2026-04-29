# Image-to-3D Generation Server

Upload a photo, get a 3D model. Progressive pipeline: fast coarse preview → async refined final.

**[Open in Colab](https://colab.research.google.com/github/srivtx/img-to-3d/blob/main/colab/Image_to_3D_Generator.ipynb)** — Free T4 GPU

---

## What It Does

1. **Upload a photo** (drag & drop or click)
2. **Get preview** in ~30-60 seconds (coarse mesh from InstantMesh)
3. **Auto-refine** in background (subdivision + smoothing)
4. **View & download** GLB files

**Tech:** FastAPI backend, Three.js frontend, InstantMesh AI, Trimesh refinement.

---

## Quick Start

### Local (Mac/Linux/Windows)

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
open http://localhost:8000
```

### Google Colab (Free T4 GPU)

Open the notebook above. Runtime → GPU. Run all cells. Get a public URL in ~5 minutes.

### Hugging Face Spaces

See `HUGGINGFACE_DEPLOY.md` for Docker deployment.

---

## Setup InstantMesh (For Real 3D)

The app works out of the box but generates mock spheres if InstantMesh isn't set up.

```bash
# 1. Clone InstantMesh code
cd models && git clone https://github.com/TencentARC/InstantMesh.git

# 2. Download weights (~4 GB)
pip install huggingface-hub
huggingface-cli download TencentARC/InstantMesh --local-dir ./instantmesh

# 3. Enable
export USE_INSTANTMESH=true

# 4. Verify
python check_setup.py
```

**Note:** The pipeline handles device auto-detection (CUDA → MPS → CPU), FP16 on CUDA, VRAM offloading, and transformers 5.x compatibility shims automatically.

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend app |
| `/generate-3d` | POST | Upload image, returns job_id |
| `/jobs/{job_id}` | GET | Poll status & get GLB URLs |
| `/health` | GET | System status |

```bash
curl -X POST "http://localhost:8000/generate-3d" -F "image=@photo.jpg"
curl "http://localhost:8000/jobs/{job_id}"
```

---

## Architecture

```
User Upload
    ↓
FastAPI (+ BackgroundTasks)
    ↓
Job Queue (asyncio, in-memory)
    ↓
Coarse Generation (InstantMesh on GPU)
    ├─ Background removal (rembg)
    ├─ Multi-view diffusion (Zero123++)
    ├─ 3D reconstruction (LRM)
    └─ Mesh extraction (FlexiCubes)
    ↓
Preview GLB available immediately
    ↓
Async Refinement (Trimesh on CPU)
    ├─ Subdivision
    ├─ Taubin smoothing
    └─ UV generation
    ↓
Final GLB
```

---

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI app, serves frontend + API
│   ├── core/                # Config, Pydantic models
│   ├── services/            # Queue, mesh processor
│   └── pipeline/            # InstantMesh + refinement
├── frontend/                # Three.js viewer + upload UI
├── models/                  # InstantMesh code + weights (created at runtime)
├── outputs/                 # Generated GLB files per job
├── colab/                   # Google Colab notebook
├── Dockerfile               # HF Spaces deployment
└── requirements.txt
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | Auto | PyTorch device: cuda / mps / cpu |
| `USE_INSTANTMESH` | false | Enable real AI inference |
| `FP16` | true | Half precision (CUDA only) |
| `KEEP_MODELS_IN_MEMORY` | true | Preload models on startup |
| `MAX_CONCURRENT_JOBS` | 2 | Parallel generation limit |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Still shows sphere" | Run `python check_setup.py`, ensure weights downloaded |
| "CUDA out of memory" | Reduce `MAX_CONCURRENT_JOBS` to 1, or use `instant-mesh-base` config |
| "MPS error on Mac" | Automatically falls back to CPU. Or `export DEVICE=cpu` |
| "URL won't open" | Make sure you're using `localhost:8000`, not `0.0.0.0` |

---

## License

MIT
