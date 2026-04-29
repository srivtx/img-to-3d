# Part 07 — Code map: which file does what

**Prerequisites:** [Part 04](part-04-jobs-states-queue-mock-vs-real.md), [Part 06](part-06-instantmesh-pipeline-stages.md).

---

## Root-level scripts (outside `app/`)

| File | Role |
|------|------|
| **`generate_colab.py`** | One-file template that **writes** `colab/Image_to_3D_Generator.ipynb` — installs deps, downloads checkpoints, configures env, tunnels. |
| **`problem_and_sol.md`** | Chronological engineer log of **what broke and how we patched** (older narrative). **[Superseded for teaching order by Part 08 + this doc series.]** |
| **`learn.md`** | Long Claude-written study guide — **mapped in** `LEARN-MD-MAP.md`. |
| **`CONTEXT.md`** | High-level project context (may lag code; trust code + Parts 06–09). |

---

## `app/main.py`

- Builds **FastAPI** app + static mount for `frontend/`.
- Registers routes including **`/`** (static **`index.html`**), **`POST /generate-3d`**, **`GET /jobs/{job_id}`**, **`GET /health`**; mounts **`/outputs`** (GLBs) and **`/static`** (frontend assets).

- **Critical:** **`async`** handlers must **not** block the event loop forever — synchronous GPU-heavy work wrapped in **`asyncio.to_thread(...)`** when needed (deep dive Part 09).

---

## `app/core/models.py`

- **Pydantic** models for HTTP: **`Generate3DResponse`**, **`JobStatusResponse`**, **`HealthResponse`**, **`JobStatus`** enum (`pending`, `processing_coarse`, …).

---

## `app/services/queue.py`

- **Dataclass** **`Job`** (in-memory): paths, **`JobStatus`**, progress %, **`error`**, timestamps.
- **`JobQueue`** with semaphore-backed concurrency caps and TTL cleanup — **no Redis.**

---


## `app/pipeline/instantmesh.py`

- Implements **`CoarseGenerator`** facade + **`run_instantmesh_pipeline()`** syncing `models/InstantMesh` layout expectations.
- **Env:** **`USE_INSTANTMESH`** flag, shim imports, optional memory tricks.
- This is **where ONNX / Diffusers / torch** exceptions most often originate before mock fallback — pay attention here when debugging silence.

---

## `app/services/mesh_processor.py`

- Applies **Gaussian smooth**, **`simplify_quadric_decimation`**, **`remove_duplicated_faces`**.
- Lesson learned: subdivide UV ops **stripped vertex colors** in some configurations — guarded paths here.

---

## `app/pipeline/refinement.py`

- Thin **`RefinementProcessor`** bridging mesh refinement stage from job orchestration layer.

---

## `frontend/` snapshot

Vanilla **`index.html` + Three.js`** — uploads file, polls job status, swaps GLB loaders + progress UI hooks.

---

**Next:** the full **human chronicle** of breakages **[Part 08](part-08-chronicle-breakages-fixes-beginner-friendly.md)** — read when you’ve got ☕️.
