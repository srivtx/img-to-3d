# Part 09 — Cross-cutting: async, VRAM, dependency shims, refinement

**Prerequisites:** [Part 08](part-08-chronicle-breakages-fixes-beginner-friendly.md).

This part ties **server design** to **ML constraints** without repeating the whole bug list.

---

## 1. FastAPI is async — your GPU code is not (usually)

**Asyncio** runs one **event loop** thread. If you **block** that thread for 60 seconds of PyTorch, **every other request** (**`/jobs/...`** polling, **`/health`**) **waits**.

### Incremental mental model

1. **Level 1:** `def blocking(): heavy()` inside `async def route()` → **bad** — blocks loop.
2. **Level 2:** `await asyncio.to_thread(blocking)` → worker thread runs heavy code; loop stays **responsive**.
3. **Level 3:** **`threading.Lock`** around GPU sections if multiple threads might touch CUDA **without** serializing unintentionally (**`learn.md` §6** pattern).

In **`app/main.py`**, **`process_job`** uses **`await asyncio.to_thread(coarse_generator.generate, ...)`** and **`await asyncio.to_thread(refinement_pipeline.refine, ...)`.**

---

## 2. Boot-time loads vs request-time loads

Startup may call **`coarse_generator.load`** — also dispatched via **`asyncio.to_thread`** when **`KEEP_MODELS_IN_MEMORY`** — so **`/health`** can answer during multi-minute weight loads.

---

## 3. VRAM tactics (incremental sophistication)

### 3a. Understand bytes

Rough formula: **`num_elements × sizeof(dtype)`**.

If shape gets **multipliers** (`grid_res`³-ish, wide feature maps), **doubling** one hyperparameter can **more than double** memory.

### 3b. Allocator option

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — reduces **fragmentation** (holes). Does **not** increase total VRAM.

### 3c. Offload / park

Move **unused** large modules **off GPU** between stages (Part 08). **Round-trip** cost usually **small** vs diffusion time.

### 3d. fp16 warnings on CPU

Moving fp16 tensors to CPU can warn — often **benign** if numerics still OK for that stage. Read whether it’s **warning** vs **exception**.

---

## 4. Dependency shims (when pinning hurts)

**Pinning** older `transformers` may break **`diffusers`**.

**Shimming** = re-add **small removed functions** at runtime **before** vendored code imports them.

**Rules of thumb:**

- Copy **minimal** code from a **known-good** old version.
- Patch **classes** when **all instances** must see a new method.
- Document **why** — future you will delete the shim when upstream fixes land.

---

## 5. Refinement: Trimesh visuals (two kinds)

**Vertex colors** vs **UV textures** are different **visual channels**.

- **Subdivide** without color interpolation → **lose** color alignment.
- Assigning **UV texture visuals** can **replace** **ColorVisuals** → **white** appearance in viewers.

**Our approach:** detect **`visual.kind`** — **vertex/face** → **Taubin** + color-safe simplify; else **geometric** pipeline with UV generation.

**API detail:** **`simplify_quadric_decimation(face_count=N)`** — **keyword** `face_count`, not a raw positional int (which Trimesh may interpret as **`percent`**).

---

## 6. Pre-flight checklist (before “it’s slow / it’s mock”)

- [ ] **`import_module`** for **`rembg`**, **`torch`**, **`diffusers`**, etc.
- [ ] Weights present under **`models/InstantMesh/...`**
- [ ] **`INSTANTMESH_CONFIG`** matches target GPU class
- [ ] Watch **`stderr`** for **`[InstantMesh]`** lines

---

**Next:** [Part 10 — Deployment: Colab vs Hugging Face vs cloud](part-10-deployment-colab-vs-huggingface-vs-cloud.md).
