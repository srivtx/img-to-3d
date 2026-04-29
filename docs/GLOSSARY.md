# Glossary — terms used across `docs/`

Terms are alphabetical. Many link to **[Part 05](part-05-ml-concepts-layer-by-layer.md)** or **[Part 08](part-08-chronicle-breakages-fixes-beginner-friendly.md)**.

---

### API (HTTP)
Rules for **browser ↔ server** messages: URLs, verbs (**GET**, **POST**), status codes (**200**, **404**). See [Part 03](part-03-web-browser-server-http.md).

### ASGI / Uvicorn
**ASGI** = standard interface for async Python servers; **Uvicorn** implements it and runs FastAPI. See Part 03.

### Async / Event loop / `asyncio.to_thread`
The **async** server uses one **loop** thread. Heavy sync code runs in **threads** (`to_thread`) so requests don’t freeze. See [Part 09](part-09-cross-cutting-async-vram-deps-refinement.md).

### Checkpoint / weights
Saved **numbers** learned during training (**`.safetensors`**, **`.bin`**, etc.). Inference **loads** them; it does **not** create them from scratch in this repo.

### Community pipeline mirror (Diffusers)
A HF dataset URL pattern Diffusers uses for **bare** `custom_pipeline` names — often **fragile**. We bypass with a **disk path**.

### Convolution / CNN (very short)
Image layers that detect **patterns** hierarchically — standard image backbone. Part 05.

### CUDA / CUDA OOM / VRAM
**CUDA**: NVIDIA GPU compute API. **VRAM**: GPU RAM. **OOM**: allocator cannot reserve a tensor — must change **config**, **batch**, offload, or hardware. Parts 05–09.

### Diffusion / diffusion steps
Generation by **denoising** latent noise (**`num_inference_steps`** trades speed vs quality). Part 05.

### DiNo / ViT
**Vision Transformer** family used inside InstantMesh’s encoder paths — hit **transformers** 5.x API removals (Part 08).

### FastAPI / Pydantic
**FastAPI**: web framework.**Pydantic**: typed request/response models (`app/core/models.py`).

### FlexiCubes / marching cubes variants
Algorithms turning **implicit fields / grids** into **triangle meshes** — VRAM-heavy at high **grid resolutions**.

### GLB / glTF
**glTF** is a portable 3D scene format.**GLB** is the binary “single-file” version (mesh + textures), easy for Three.js.

### Hugging Face Hub
Hosts models & datasets; **`from_pretrained`** downloads — when paths are known.

### Job / polling
**Upload returns `job_id` immediately.** Client **polls** `GET /jobs/{id}` until **terminal** status. Part 04.

### Mock (icosphere) vs real
If loading fails or env disables ML, **`CoarseGenerator`** can serve a placeholder **sphere mesh** so the plumbing still runs. Danger: looks “fine”. Part 04 + 08.

### Multi-view synthesis
Produce **novel RGB views** of an object consistent with input — enables 3D fusion. Part 05–06.

### ONNX Runtime
Runtime for `.onnx` graphs; pulled in by **`rembg`** segmentation path. Missing install → **`import rembg` fails.**

### Ports (`8000`)
A **logical slot** where the server listens. Part 03.

### Pre-flight smoke test
**Actually import** critical modules (`import_module`) **before** long-running server subprocess. Distinct from **`find_spec`**.

### Shim (compatibility shim)
Tiny code re-inserting removed library symbols so **vendored** research code survives **new dependency versions.**

### Transformer (architecture — not Hugging Face)
Attention-based architecture; **⚠️** don’t confuse with **`transformers` library** noun only.

### **`transformers` (library)**
Hugging Face NLP/CV backbone library.**5.x removed** helpers InstantMesh relied on (**Part 08**).

### Trimesh
Python mesh toolkit — smoothing, decimation (**`simplify_quadric_decimation`**), GLB export. Refinement quirks: **visuals**.

### Tunnel (Cloudflare, ngrok)
Public **HTTPS** URL forwarded to **`localhost`** on a remote VM (**Colab**). Part 10.

### Vertex colors vs UV texture
Different ways to **paint** meshes — changing **visual** type carelessly loses colors (**Part 09**).

---

**Teaching order:** prefer [README reading table](README.md) over alphabetical lookup.
