# Part 4: Our Architecture — What We Built

## Philosophy Recap

We didn't start with AI. We started with a **button that shows a sphere.**

Why? Because if the button works, the upload works, the progress bar works, the 3D viewer works, and the download works — then replacing the sphere with real 3D is just **one more step**.

This is our architecture: **layers of working systems, stacked incrementally.**

---

## Layer 0: The Mock (Where We Started)

Before any AI, our app did this:

```
User uploads photo
    ↓
Server creates a sphere (instant, no AI)
    ↓
Server "refines" the sphere (smooths it, still a sphere)
    ↓
User sees a sphere in 3D viewer
    ↓
User downloads sphere.glb
```

**Why this is valuable:**
- The entire UX works
- The API contract is defined
- The frontend knows what to expect
- We can test end-to-end without waiting for AI

**The sphere is our scaffold.** We built the cathedral around it, then replaced the scaffold.

---

## Layer 1: The Frontend (What Users See)

### Technology
- **HTML/CSS:** Structure and styling
- **Vanilla JavaScript:** Logic (no frameworks)
- **Three.js:** 3D rendering in the browser

### Components

#### Upload Zone
```
┌─────────────────────────────┐
│         [Icon]              │
│   Drop image or click       │
│   JPG, PNG, WebP            │
└─────────────────────────────┘
```

**How it works:**
1. A hidden `<input type="file">` covers the whole area
2. When you click anywhere, the file picker opens
3. When you drop a file, JavaScript catches the `drop` event
4. The file is validated (type, size)

**Why we did it this way:** Native file inputs are reliable. Custom drag-and-drop UIs often break. By making the native input invisible but clickable, we get both: custom styling AND browser reliability.

#### Status Card
```
┌─────────────────────────────┐
│  PROCESSING      45%        │
│  ████████████░░░░░░░░       │
│  Generating views...        │
└─────────────────────────────┘
```

**How it works:**
1. After upload, JavaScript starts a timer
2. Every second: `GET /jobs/{job_id}`
3. Updates progress bar and status text
4. When status = "coarse_ready", loads preview model
5. When status = "completed", loads final model

**Why polling?** It's simple, works everywhere, and doesn't require WebSocket infrastructure. For our use case (updates every second), polling is fine.

#### 3D Viewer
```
┌─────────────────────────────┐
│                             │
│     [3D Model Here]         │
│                             │
│  ← Drag to rotate           │
│  ← Scroll to zoom           │
└─────────────────────────────┘
```

**How Three.js works:**
1. Creates a WebGL canvas
2. Sets up scene, camera, lights
3. Loads GLB file via `GLTFLoader`
4. Centers and scales the model
5. Renders 60 frames per second
6. OrbitControls handles mouse input

**The render loop:**
```javascript
function animate() {
    requestAnimationFrame(animate)  // Call again next frame
    controls.update()                // Handle mouse input
    renderer.render(scene, camera)   // Draw everything
}
```

**Why WebGL?** It's the only way to render 3D in browsers at good performance. Three.js wraps WebGL in an easier API.

---

## Layer 2: The Backend (The Server)

### Technology
- **FastAPI:** Python web framework
- **Uvicorn:** ASGI server (runs FastAPI)
- **Pydantic:** Data validation

### File Structure

```
app/
├── main.py              # Entry point, routes
├── core/
│   ├── config.py        # Settings (DEVICE, FP16, etc.)
│   └── models.py        # Pydantic request/response schemas
├── services/
│   ├── queue.py         # Job queue
│   └── mesh_processor.py # Mesh refinement
└── pipeline/
    ├── instantmesh.py   # AI model wrapper
    └── refinement.py    # Post-processing
```

### The Routes

#### POST /generate-3d
**What it does:**
1. Validate image file (type, size)
2. Save to disk
3. Create job in queue
4. Start background task
5. Return job_id immediately

**Why return immediately?** AI takes 30-60 seconds. Users shouldn't wait. They get a ticket (job_id) and check later.

```python
@app.post("/generate-3d")
async def generate_3d(image: UploadFile):
    job = await queue.create_job()
    save_image(image, job.id)
    background_tasks.add_task(process_job, job.id)
    return {"job_id": job.id, "status": "pending"}
```

#### GET /jobs/{job_id}
**What it does:**
1. Look up job in queue
2. Return current status
3. Include preview/final URLs if available

**Why polling?** Simple, stateless, works through firewalls.

```python
@app.get("/jobs/{job_id}")
async def get_status(job_id: str):
    job = await queue.get_job(job_id)
    return {
        "status": job.status,
        "progress": job.progress_percent,
        "preview_url": job.preview_url,
        "final_url": job.final_url
    }
```

#### GET /health
**What it does:**
1. Check if server is running
2. Check if models are loaded
3. Report queue stats

**Why?** Monitoring tools need a simple "are you alive?" endpoint.

---

## Layer 3: The Job Queue

### Why We Need a Queue

When user A uploads a photo, AI takes 30 seconds. While processing:
- User B shouldn't get "server busy"
- User A shouldn't have their browser hang
- The API should keep responding

**Solution:** Queue + Background Tasks

### How It Works

```
Upload → Create Job (PENDING) → Add to Queue → Return Job ID
                                           ↓
Background Worker → Pick Job → PROCESSING → Run AI → COMPLETED
```

### Job States

| State | Meaning | What User Sees |
|-------|---------|----------------|
| **PENDING** | Waiting to start | "Queued..." |
| **PROCESSING_COARSE** | Running AI | "Generating 3D model..." |
| **COARSE_READY** | Preview done | "Preview ready! Refining..." |
| **REFINING** | Post-processing | "Improving quality..." |
| **COMPLETED** | Done | "Done! Download your model" |
| **FAILED** | Error | "Something went wrong" |

### Implementation

Our queue is **in-memory** (simple dictionary + asyncio locks).

**Why in-memory?**
- Simple (no Redis/Celery setup)
- Fast (no network calls)
- Good enough for single-server deployments

**Trade-off:** Jobs disappear if server restarts. For production, you'd use Redis.

```python
class JobQueue:
    def __init__(self):
        self.jobs = {}  # job_id -> Job object
        self.lock = asyncio.Lock()
    
    async def create_job(self):
        job = Job(id=uuid4(), status=PENDING)
        async with self.lock:
            self.jobs[job.id] = job
        return job
```

---

## Layer 4: The AI Pipeline

### Stage 1: Coarse Generation (InstantMesh)

**Input:** Single RGB image (your photo)
**Output:** 3D mesh file (preview.glb)
**Time:** ~30-60 seconds on T4 GPU

#### What InstantMesh Actually Does

```
Your Photo
    ↓
[Remove Background] (optional, via rembg)
    ↓
[Zero123++ Diffusion] → Generate 6 views from different angles
    ↓
[Reconstruction Model] → Fuse 6 views into 3D geometry
    ↓
[FlexiCubes Extraction] → Convert implicit field to triangle mesh
    ↓
[Export] → Save as GLB
```

#### Step-by-Step

**1. Background Removal (rembg)**
- Uses a neural network (U2Net) to segment foreground from background
- Why? The 3D model should only contain the object, not the room behind it
- Output: RGBA image (alpha channel = transparency)

**2. Multi-View Generation (Zero123++)**
- Diffusion model generates 6 consistent views
- Each view is 320×320 pixels
- Arranged in a 3×2 grid (front, right, back, left, top, bottom)
- **Key:** These views are consistent — they show the same object from different angles

**3. Reconstruction (LRM — Large Reconstruction Model)**
- Takes 6 views + camera parameters
- Outputs a **triplane** representation (3 orthogonal feature planes)
- The triplane encodes the 3D shape implicitly

**4. Mesh Extraction (FlexiCubes)**
- Converts the implicit triplane into explicit triangles
- Uses differentiable marching cubes variant
- Outputs: vertices, faces, vertex colors

**5. Export to GLB**
- Reorients axes (Y-up for glTF)
- Saves as binary GLB file

#### VRAM Management

The T4 GPU has 16 GB. The pipeline needs:
- Diffusion model: ~3 GB
- Reconstruction model: ~2 GB
- Intermediate tensors: ~6 GB
- **Total peak:** ~11 GB

**Strategy:** After diffusion, offload the pipeline to CPU:
```python
pipeline.to("cpu")  # Free 3 GB
torch.cuda.empty_cache()
# Run reconstruction with freed memory
```

Next request moves it back to GPU (1-2 second cost).

### Stage 2: Refinement

**Input:** preview.glb (coarse mesh)
**Output:** final.glb (smoothed mesh)
**Time:** ~5-10 seconds on CPU

#### What Refinement Does

```python
def refine_mesh(input_path, output_path):
    mesh = trimesh.load(input_path)
    
    # 1. Subdivide (add more triangles)
    mesh = mesh.subdivide()
    
    # 2. Smooth (Taubin smoothing)
    mesh = smooth_mesh(mesh)
    
    # 3. Ensure UVs exist
    mesh = add_uvs(mesh)
    
    mesh.export(output_path)
```

**Why refinement?**
- InstantMesh output is optimized for speed, not beauty
- Subdivision adds geometric detail
- Smoothing removes jagged edges
- UVs ensure textures can be applied later

**Why CPU?** Refinement is lightweight. GPU isn't needed.

---

## Layer 5: Configuration & Deployment

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `DEVICE` | cuda / mps / cpu |
| `USE_INSTANTMESH` | Enable real AI (false = mock) |
| `FP16` | Half precision (faster, less VRAM) |
| `KEEP_MODELS_IN_MEMORY` | Preload models at startup |
| `MAX_CONCURRENT_JOBS` | Parallel generation limit |
| `INSTANTMESH_CONFIG` | Model size (base/large) |
| `INSTANTMESH_DIFFUSION_STEPS` | Quality vs speed tradeoff |

### Auto-Detection

```python
if torch.cuda.is_available():
    DEVICE = "cuda"      # NVIDIA GPU
elif torch.backends.mps.is_available():
    DEVICE = "mps"       # Apple Silicon
else:
    DEVICE = "cpu"       # Fallback
```

### Deployment Options

| Platform | Cost | GPU | Persistent? | Best For |
|----------|------|-----|-------------|----------|
| **Local/Mac** | Free | MPS/CPU | Yes | Development |
| **Colab T4** | Free | T4 16GB | No (12hr max) | Testing |
| **HF Spaces CPU** | Free | No | Yes (sleeps) | Demos |
| **HF Spaces GPU** | ~$0.50/hr | L4 | Yes (sleeps) | Production |
| **RunPod** | ~$0.30/hr | 3090/A100 | Yes | Production |

---

## The Full Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER'S BROWSER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Upload Zone  │  │ Status Card  │  │   3D Viewer          │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼─────────────────┼─────────────────────┼──────────────┘
          │ POST /generate-3d   GET /jobs/{id}      GET /outputs/*.glb
          │                     (poll every 1s)     (load model)
          ↓                     ↓                   ↓
┌─────────────────────────────────────────────────────────────────┐
│                        FASTAPI SERVER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Upload API  │→ │   Queue      │→ │   Job Processor      │  │
│  └──────────────┘  └──────────────┘  └──────────┬───────────┘  │
│                                                 │                │
│  ┌──────────────┐  ┌──────────────┐            │                │
│  │  Health API  │  │  Static Files│            │                │
│  └──────────────┘  └──────────────┘            ↓                │
│                                       ┌──────────────────┐     │
│                                       │ Coarse Generator │     │
│                                       │ (InstantMesh)    │     │
│                                       └────────┬─────────┘     │
│                                                │                │
│                                       ┌────────▼─────────┐     │
│                                       │ Mesh Refinement  │     │
│                                       │ (Trimesh)        │     │
│                                       └──────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
          ↑
          │ Serve frontend (index.html, app.js, style.css)
          │ Serve outputs (preview.glb, final.glb)
```

---

## Design Decisions Explained

### Why FastAPI over Flask/Django?

- **Native async:** Handles many concurrent uploads without blocking
- **Auto-generated docs:** `/docs` gives you an interactive API explorer
- **Type hints:** Catches bugs before runtime
- **Performance:** Faster than Flask for I/O-bound workloads

### Why polling over WebSockets?

- **Simpler:** No connection management
- **More reliable:** Works through corporate firewalls
- **Scalable:** Stateless, easy to load balance
- **Good enough:** 1-second updates are fine for 30-60 second jobs

### Why mock mode?

- **Development speed:** UI team can work without AI setup
- **Testing:** Verify pipeline without waiting for GPU
- **Fallback:** If AI fails, user still gets a response
- **Demo:** Show the product before AI is ready

### Why GLB over OBJ?

- **Single file:** Easy to download and share
- **Web-native:** Three.js loads it directly
- **Binary:** Smaller than text formats
- **Standard:** Supported by Blender, Unity, Unreal

---

## Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | HTML/CSS/JS + Three.js | Upload UI, 3D viewer, progress |
| **Backend** | FastAPI + Uvicorn | API routes, file serving |
| **Queue** | In-memory asyncio | Job tracking, async processing |
| **Coarse AI** | InstantMesh (PyTorch) | Photo → 3D mesh |
| **Refinement** | Trimesh | Mesh smoothing, subdivision |
| **Storage** | Local filesystem | Images, GLB files |

---

**Next:** [Part 5: Everything That Broke](part-05-everything-that-broke.md) — Our chronological saga of errors, fixes, and lessons.
