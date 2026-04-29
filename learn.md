# Lessons Learned — Wiring an ML Inference Pipeline

A reference doc collecting everything we learned while turning the
image-to-3D project from "falls back to a mock icosphere" into "produces
a real textured 3D mesh on Colab GPU." Written so you can read individual
sections without context — each one stands on its own.

The chronological bug log is in [`problem_and_sol.md`](problem_and_sol.md).
This doc is the *generalized* version: patterns, mental models, and
reusable techniques you can apply to other ML inference projects.

---

## Table of Contents

1. [Debugging strategy](#1-debugging-strategy)
2. [The Python import system](#2-the-python-import-system)
3. [Library version drift](#3-library-version-drift)
4. [GPU memory management](#4-gpu-memory-management)
5. [Trimesh visual system](#5-trimesh-visual-system)
6. [FastAPI for ML](#6-fastapi-for-ml)
7. [Colab / notebook pipeline](#7-colab--notebook-pipeline)
8. [The InstantMesh architecture](#8-the-instantmesh-architecture)
9. [Operational patterns](#9-operational-patterns)
10. [General software lessons](#10-general-software-lessons)

---

## 1. Debugging strategy

### 1.1 The "deepest visible failure" pattern

When you fix one bug, the next bug is the *next thing in the call chain
that was waiting to fail*. Each fix doesn't bring you closer to "done"
in some smooth way — it advances you to a new, deeper failure that was
previously hidden behind the earlier one. We hit this seven times in a
row on this project:

1. `import rembg` → failed at `import onnxruntime`
2. Fix onnxruntime → next failure: `from transformers.pytorch_utils import find_pruneable_heads_and_indices`
3. Fix shim → next: pipeline not found at community-pipelines-mirror URL
4. Point to vendored pipeline.py → next: `'ViTModel' has no attribute 'get_head_mask'`
5. Shim get_head_mask → next: CUDA OOM in mesh extraction
6. Switch to instant-mesh-base + offload → next: refinement failed (`fast_simplification` missing)
7. Install fast_simplification → next: `final.glb` lost vertex colors

The mental model:

> Each bug is a "domino." Fixing it doesn't *reveal* progress; it just
> reveals the next domino. The fact that progress is happening is
> measurable by *how far the failure point has moved* in the call
> stack, not by how few errors you're seeing.

A useful rule of thumb: when a failure happens in a *different* file
than the previous one, and *deeper* in the call chain (e.g. inference
time vs load time, or stage 3 vs stage 1), you're winning even if the
log still says ERROR.

### 1.2 Buffered stdout hides everything

Python's `print()` is **block-buffered** when stdout is redirected to
a non-terminal (file, pipe, etc.). For our server, that meant:

```python
server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", ...],
    stdout=server_log_file,  # redirected -> block buffered
    stderr=subprocess.STDOUT,
)
```

Inside the server, our `print(f"[InstantMesh] Real load FAILED -> {e}")`
was sitting in a 4 KB buffer that didn't flush until 4 KB had
accumulated *or* the process exited. The user only saw older lines.
The actual failure was invisible.

**Three fixes for this pattern:**

```python
# 1. Per-process: set PYTHONUNBUFFERED in the env when starting the
#    subprocess
env["PYTHONUNBUFFERED"] = "1"

# 2. Per-line: pass flush=True to print
print("important message", flush=True)

# 3. Per-stream: line-buffer the file you redirect to
log_file = open("/tmp/server.log", "w", buffering=1)
```

We used (3) for the Colab server cell. Buffer mode `1` means line-buffered,
so every newline triggers a flush.

**The deeper lesson:** if a subprocess seems to "decide" to do something
silent (like fall back to mock mode), suspect buffered output. Run the
load path in a *foreground* cell to bypass the redirect entirely:

```python
# Ad-hoc diagnostic cell — bypasses subprocess buffering
import sys, os
sys.path.insert(0, "/content/img-to-3d")
os.environ["USE_INSTANTMESH"] = "true"
from app.pipeline.instantmesh import coarse_generator
coarse_generator.load()  # any traceback prints directly here
```

### 1.3 Broad except clauses are anti-debugging

```python
try:
    self._load_real_model()
except Exception as e:
    print(f"FAILED: {e}")
    self._using_mock = True
```

This is the original code. It does three bad things:
1. **Catches `Exception`** — including unrelated bugs like typos, type errors, and import errors that aren't related to the model.
2. **Prints only `e`** — not the traceback. So you see "FAILED: No module named 'onnxruntime'" but not which line caused it. For multi-step failures (rembg failing inside our load path), you can't distinguish "rembg's import failed" from "we passed a bad arg to rembg".
3. **Demotes silently** — sets `_using_mock = True` so subsequent calls *continue working*, but with wrong behavior. The user sees icospheres and assumes things are fine.

**Better pattern when you want fallback behavior:**

```python
import traceback

try:
    self._load_real_model()
except Exception:
    print("[InstantMesh] Real load FAILED -> falling back to MOCK", flush=True)
    traceback.print_exc()      # full traceback so you can debug
    self._using_mock = True
    self._load_error = traceback.format_exc()  # store for /health endpoint
```

And expose `_load_error` via your `/health` endpoint so an external
observer can tell why the system is in degraded mode without reading
log files.

### 1.4 Pre-flight checks done right

We had a pre-flight check using `importlib.util.find_spec("rembg")`. It
returned a non-None spec, so we said "rembg is installed." But
`import rembg` was actually broken because rembg's `__init__.py` does
`import onnxruntime`, which was missing.

**The two-line lesson:**

```python
# WRONG -- only verifies the package directory exists on disk:
if importlib.util.find_spec("rembg") is None: ...

# RIGHT -- actually executes the package's __init__.py:
try:
    importlib.import_module("rembg")
except Exception as e:
    print(f"rembg is BROKEN: {type(e).__name__}: {e}")
```

`find_spec` is for "is this module discoverable?" — useful for optional
dependencies you'll only conditionally import later. It's the **wrong**
tool for "will my code that imports this actually run?". Always
`import_module` if you want to verify it works.

---

## 2. The Python import system

### 2.1 `find_spec` vs `import_module` vs `__import__`

| API | What it does | When to use |
|---|---|---|
| `importlib.util.find_spec("foo")` | Looks for `foo` on `sys.path`. Returns spec or None. **Does not execute `__init__.py`.** | Optional-dep detection, plugin discovery |
| `importlib.import_module("foo")` | Full import: locates AND executes `__init__.py`. Returns the module object. | Verifying a dependency works |
| `__import__("foo")` | Same as above but lower-level (used by the `import` statement). Avoid in user code. | Never — use `import_module` |

If you've ever wondered why `find_spec` is so much faster than
`import_module`: it's because it only does the cheap part (path
resolution), skipping the expensive part (running every module's
top-level code). That speed comes at the cost of not catching
`ImportError`, `SyntaxError`, or any side-effect failure inside the
imported module.

### 2.2 Function-level imports for graceful failure

Where you put your imports affects how failures propagate.

```python
# Module-level -- crashes the entire program at import time
import nvdiffrast  # if missing, your whole server won't even start

class CoarseGenerator:
    def load(self):
        ...
```

vs.

```python
# Function-level -- contained inside a try/except in the caller
class CoarseGenerator:
    def load(self):
        try:
            self._load_real_model()
        except Exception:
            self._using_mock = True

    def _load_real_model(self):
        import nvdiffrast  # only fails if real model is being loaded
        ...
```

Both styles are valid. The right choice depends on:

- **Module-level** when the dep is fundamental and the program can't do
  anything useful without it (e.g. fastapi for a fastapi-based service).
- **Function-level** when the dep is for an optional feature that the
  rest of the program can degrade gracefully without (e.g. CUDA-only
  libraries on a CPU-only deployment).

We chose function-level for InstantMesh's heavy deps (`nvdiffrast`,
`rembg`, `pytorch_lightning`, etc.) because we want the FastAPI server
to start successfully on a developer's CPU-only Mac and just fall back
to mock mode for those endpoints — without forcing them to install CUDA.

### 2.3 Adjusting `sys.path` for vendored code

InstantMesh's source code uses imports like `from src.utils.train_util
import instantiate_from_config`. That `src.` prefix only resolves if
the `models/InstantMesh/` directory is on `sys.path`:

```python
INSTANTMESH_CODE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "InstantMesh"
if str(INSTANTMESH_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(INSTANTMESH_CODE_DIR))

from src.utils.train_util import instantiate_from_config  # now works
```

Two gotchas:

1. **Use `str(path)`** — `sys.path` is a list of strings, not Path objects.
2. **Insert at position 0**, not append. Otherwise an installed package
   with the same name (e.g. some other `src` on PyPI) will shadow the
   vendored copy.

### 2.4 Class-level monkey-patching propagates to existing instances

When we shimmed `transformers.PreTrainedModel.get_head_mask` *after*
the model was already loaded, did the existing model instance pick up
the new method? **Yes.**

```python
# Time T1: load model
model = ViTModel(...)  # instance created

# Time T2: monkey-patch the class
PreTrainedModel.get_head_mask = lambda self, ...: ...

# Time T3: call method on existing instance
model.get_head_mask(...)  # uses the patched version!
```

Why: when you call `instance.method(...)`, Python doesn't look up
`method` on the instance's `__dict__` first. It looks at
`type(instance).__mro__` (the method resolution order) and finds the
attribute on the class. So patching the class *immediately* affects
all existing and future instances.

**Caveat:** this only works for normal method lookup. If a method was
**bound to an instance** (e.g. via `functools.partialmethod` or stored
in `instance.__dict__`), then per-instance lookup *would* shadow the
class. Rare, but worth knowing.

---

## 3. Library version drift

### 3.1 Pinning vs shimming

Two ways to handle "library X removed an API that library Y depends on":

**Option A: Pin to old version**
```
transformers<5.0
```

Pro: zero code change.
Con: forces a 200 MB pip operation, risks breaking *other* libraries
that pin minimum versions of transformers. Also, the old version might
have *its own* compatibility issues with your modern stack (CUDA
versions, pytorch versions, etc.).

**Option B: Shim the missing API**
```python
def _shim_transformers_compat():
    if not hasattr(PreTrainedModel, "get_head_mask"):
        PreTrainedModel.get_head_mask = lambda self, ...: ...
```

Pro: works on any version, no pip dance, no dep conflicts. Cheap.
Con: code change, you have to actually understand and reimplement what
the missing API does.

**Decision matrix:**

| Situation | Choice |
|---|---|
| The missing API is small and pure-functional | Shim |
| The missing API has complex implementation | Pin |
| The dep graph is fragile | Shim |
| You control every dep | Pin |
| The "removed API" was internal/private (no semver promise) | Shim |
| The "removed API" was public and the library has good migration docs | Pin to last version supporting it |

We chose shimming for transformers 5.x removals because (a) the
removed APIs were small (15-line pure-PyTorch helpers, lifted verbatim
from transformers v4.46.0), (b) downgrading transformers would have
dragged diffusers / accelerate / huggingface-hub into a version
solving problem.

### 3.2 The "version-pinned mirror" trap

Diffusers' `custom_pipeline` resolution looks like a clever idea until
you trip on it:

```python
DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="zero123plus",  # bare name, no slash
    torch_dtype=torch.float16,
)
```

Diffusers takes that bare name and builds a URL like:

```
https://huggingface.co/datasets/diffusers/community-pipelines-mirror/resolve/main/v{your_diffusers_version}/{name}.py
```

So with diffusers 0.37.1 it tries:

```
https://huggingface.co/datasets/.../v0.37.1/zero123plus.py
```

Which 404s, because nobody at Hugging Face publishes every community
pipeline file at every version of diffusers. The community pipelines
mirror is curated. Files come and go between versions.

**Three resolution modes in `custom_pipeline`:**

```python
# 1. Bare name (foo) -> mirror lookup, FRAGILE
custom_pipeline="zero123plus"

# 2. Slash-separated (user/repo) -> HF Hub repo, looks for pipeline.py at root
custom_pipeline="sudo-ai/zero123plus-v1.2"

# 3. Absolute path on disk -> loads file directly, MOST RELIABLE
custom_pipeline="/content/img-to-3d/models/InstantMesh/zero123plus/pipeline.py"
```

**Lesson:** for production code, prefer (3) over (1). The model author
usually vendors `pipeline.py` somewhere stable — find it and point
directly at it. The mirror lookup is convenient but brittle.

### 3.3 Optional pip extras silently skipping deps

`rembg` is a perfect case study. As of `rembg>=2.0.50`, `onnxruntime` is
declared as an optional extra:

```toml
[project.optional-dependencies]
cpu = ["onnxruntime"]
gpu = ["onnxruntime-gpu"]
```

So `pip install rembg` installs the library but **not** onnxruntime.
The user has to know to do `pip install rembg[cpu]` or
`pip install rembg onnxruntime` to get a working install.

But rembg's `__init__.py` does `import onnxruntime` unconditionally.
So the install "succeeds" but the import fails. Pure footgun.

**How to detect this in your own dependency lists:**

When you take a new dep, do:

```bash
python -c "import the_dep; print('OK')"
```

after pip install. If that crashes, you have an undeclared transitive
dep. Add the missing package to your install list explicitly.

---

## 4. GPU memory management

### 4.1 VRAM math

When `torch.OutOfMemoryError` says:

```
Tried to allocate 15.00 GiB.
GPU 0 has a total capacity of 14.56 GiB of which 6.91 GiB is free.
Including non-PyTorch memory, this process has 7.65 GiB memory in use.
Of the allocated memory 7.23 GiB is allocated by PyTorch,
and 301.94 MiB is reserved by PyTorch but unallocated.
```

Read this systematically:

| Number | Meaning |
|---|---|
| 14.56 GiB | Hardware ceiling (GPU's physical VRAM, minus driver reserve) |
| 7.65 GiB | Currently used by *all* processes on this GPU (could include other notebooks) |
| 7.23 GiB | Currently used by *this* PyTorch process |
| 6.91 GiB | Free for new allocations |
| 301.94 MiB | Cached by PyTorch but not handed to live tensors (fragmentation slack) |

If `Tried to allocate > free + slack` → you're memory-pressed but might
be saved by `empty_cache()`.

If `Tried to allocate > total - non_pytorch_memory` → you need to evict
other processes/tensors.

**If `Tried to allocate > total capacity`** → no amount of freeing
helps. This single tensor cannot fit on this GPU. Change the model
config (smaller variant, lower resolution) or change the GPU.

The 15 GB allocation we hit was this last case: 15 > 14.56. Math
problem, not memory-management problem.

### 4.2 Knowing why a tensor is X bytes

Tensor size in bytes = `(product of dims) × dtype_size_in_bytes`.

| dtype | bytes per element |
|---|---|
| float32 / int32 | 4 |
| float16 / bfloat16 / int16 | 2 |
| int8 / uint8 / float8 | 1 |
| float64 / int64 | 8 |

Our 15 GB allocation was inside InstantMesh's `synthesizer_mesh.py`:

```python
grid_features = torch.index_select(
    input=sampled_features,      # [1, V, F]
    index=flexicubes_indices.reshape(-1),  # ~16M indices
    dim=1,
)
# Output shape: [1, 16M, F]
# At F=256 (instant-mesh-large): 1 × 16M × 256 × 4 bytes = 16 GB
# At F=128 (instant-mesh-base):  1 × 16M × 128 × 4 bytes =  8 GB
```

When you OOM, do this size math. If the answer is "this is bigger
than my GPU," changing the config is the only fix — code-level
optimization can't help.

### 4.3 The `expandable_segments` allocator

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

By default, PyTorch's CUDA allocator preallocates fixed-size segments.
If you allocate a big tensor, free it, then allocate a *different-sized*
big tensor, the second allocation might fail even though there's
technically enough free memory — because the freed memory is in
fixed-size chunks that don't match the new request. This is
fragmentation.

`expandable_segments:True` switches to a more flexible scheme that can
combine adjacent free chunks. Costs slightly more bookkeeping, but
prevents most fragmentation OOMs.

**Rule of thumb:** if you're seeing OOMs *after* successful allocations
have completed and freed (i.e., later in a session, not on the very
first attempt), set this. We did, and it removed several spurious
OOMs.

### 4.4 Pipeline offloading: park, don't run

Standard pattern for swapping models in/out of VRAM:

```python
# Free the pipeline's VRAM between uses
self.pipeline.to("cpu")
torch.cuda.empty_cache()

# ... do other GPU work that needs the freed memory ...

# Bring it back when needed
self.pipeline.to(self.device)
```

This is "parking" — you move the model to CPU memory, NOT to run it
there, just to get it out of the way. Bring it back to GPU before any
forward pass.

Round-trip cost on T4:
- ~3 GB model: ~1-2 seconds CPU↔GPU each direction
- Negligible if your GPU work between offloads is more than 5 seconds

**Important:** if your pipeline was loaded with `dtype=torch.float16`,
diffusers will warn ("cannot run with cpu device") when you `.to("cpu")`.
That warning is correct — running fp16 on CPU is not supported — but
it doesn't apply to *parking*. Suppress the warning if it bothers you;
it's not an error.

### 4.5 fp16 + CPU = warning, not error

PyTorch's CPU implementation has limited fp16 support:
- Tensor *creation* and *device transfer* are fine
- Most *operations* (matmul, conv, etc.) either crash or fall back to
  software-emulated math

So `pipeline.to("cpu")` works. `pipeline(image)` while on CPU would
fail. As long as you only call the pipeline while it's on GPU, fp16 is
fine.

If you actually need to run on CPU (e.g. on a laptop without a GPU),
load the pipeline with `torch.float32` instead.

---

## 5. Trimesh visual system

Trimesh has three classes for "what does this mesh look like":

```python
from trimesh.visual.color import ColorVisuals
from trimesh.visual.texture import TextureVisuals
# (and there's a base class `Visual`)
```

### 5.1 `ColorVisuals` — for vertex/face colors

Stores `vertex_colors` (one RGBA per vertex) or `face_colors` (one RGBA
per face). The `kind` property tells you which:

```python
mesh.visual.kind  # "vertex" | "face" | None
```

`None` means there's a ColorVisuals object but no real color data —
Trimesh defaults to white. Always check `kind in ("vertex", "face")`
to know if there's *real* color, not just a placeholder.

### 5.2 `TextureVisuals` — for UV-mapped meshes

Stores `uv` (one (u,v) per vertex), a `material` (with diffuse texture,
normal map, etc.), and the texture image bytes. **Has no concept of
vertex_colors.**

### 5.3 The destructive UV assignment

This is the gotcha that destroyed our `final.glb` colors:

```python
mesh.visual.uv = np.stack([u, v], axis=-1)
```

On a mesh that previously had `ColorVisuals`, this assignment **replaces
the visual with `TextureVisuals`** — even though syntactically you only
seem to be setting one attribute. The vertex_colors are gone forever
after this line.

**The right defensive pattern:**

```python
def assign_uvs_safely(mesh, uv):
    if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        # Don't override -- would lose vertex_colors
        return mesh
    if mesh.visual.kind in ("vertex", "face"):
        return mesh
    mesh.visual.uv = uv
    return mesh
```

### 5.4 Operations that drop visual data

Trimesh operations vary in how they handle `mesh.visual`:

| Operation | Preserves vertex_colors? |
|---|---|
| `mesh.subdivide()` | **No** (creates new vertices with no color) |
| `mesh.simplify_quadric_decimation()` | **No** (delegates to fast_simplification, which doesn't carry attributes by default) |
| `trimesh.smoothing.filter_taubin(mesh)` | **Yes** (in-place vertex move, doesn't touch visuals) |
| `trimesh.smoothing.filter_laplacian(mesh)` | **Yes** (same as Taubin) |
| `mesh.copy()` | **Yes** |
| `mesh.merge_vertices()` | Mostly yes |
| `mesh.export("file.glb")` then `trimesh.load(...)` | Yes, IF saved with vertex_colors |

**Lesson:** for any mesh op that creates new vertices (subdivide,
simplify, remesh), assume colors are LOST unless you've verified
otherwise. If you need to preserve them through such an op, you'll
have to interpolate manually using `mesh.kdtree` or
`mesh.nearest.vertex(...)`.

### 5.5 GLB export semantics

`mesh.export("file.glb")`:
- If `mesh.visual` is `ColorVisuals` with vertex_colors → embeds them
  as a vertex attribute. Three.js renders them natively.
- If `TextureVisuals` with uv + material → embeds the texture image
  bytes and UV mapping.
- If neither → exports a flat-shaded mesh (default white in viewers).

There's no "both" mode in glTF 2.0. A mesh has either vertex colors
OR a textured material, not both.

---

## 6. FastAPI for ML

### 6.1 The async event loop blocking problem

FastAPI is async by default. Endpoints look like:

```python
@app.post("/generate-3d")
async def generate(image: UploadFile):
    job_id = await queue.enqueue(image)
    return {"job_id": job_id}
```

This works as long as everything inside the endpoint is fast or
properly awaited. But ML inference is **synchronous and slow**:

```python
async def process_job(job_id):
    # ... 50 seconds of GPU work ...
    coarse_generator.generate(image_path)  # BLOCKS the event loop!
```

While `generate()` is running, **no other async code can execute** —
not other endpoints, not status polling, nothing. The whole server
becomes single-threaded for the duration.

### 6.2 The fix: `asyncio.to_thread`

```python
import asyncio

async def process_job(job_id):
    preview_path, _ = await asyncio.to_thread(
        coarse_generator.generate,  # synchronous, blocking
        image_path,
        output_dir,
        num_views,
    )
```

`asyncio.to_thread` runs the synchronous function in a thread pool,
returning a Future that the event loop can await without blocking.
Other endpoints (especially `/jobs/<id>` polling) keep working
during the 50-second inference.

### 6.3 GPU serialization with `threading.Lock`

But wait — if N async requests all call `asyncio.to_thread(generate)`
simultaneously, you'll have N threads trying to use the GPU at once,
which OOMs immediately.

Solution: a `threading.Lock` inside the generator:

```python
class CoarseGenerator:
    def __init__(self):
        self._gpu_lock = threading.Lock()

    def generate(self, image_path, output_dir, num_views):
        with self._gpu_lock:  # serializes GPU access
            return self._real_generate(image_path, output_dir, num_views)
```

Now even if 10 requests fire simultaneously, only one runs at a time
on the GPU. The others wait. Async event loop stays responsive
(threads are blocked on the lock, not the loop).

### 6.4 Health endpoints for verification

Always expose enough info via `/health` to debug from outside:

```python
@app.get("/health")
def health():
    return {
        "status": "healthy" if ready else "degraded",
        "models_loaded": coarse_generator._loaded,
        "using_mock": coarse_generator._using_mock,  # critical
        "device": str(coarse_generator.device),
        "queued_jobs": queue.size(),
    }
```

The `using_mock` field saved us hours of debugging. Without it, you
can't tell from outside whether the server is producing real or fake
output.

---

## 7. Colab / notebook pipeline

### 7.1 Generating notebooks programmatically

`.ipynb` files are JSON. Hand-editing them is brittle — escape sequences
break, cell indices shift, JSON gets corrupted on accident.

**Use `nbformat`:**

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell("# Step 1: Install"))
nb.cells.append(nbf.v4.new_code_cell("!pip install ..."))

with open("notebook.ipynb", "w") as f:
    nbf.write(nb, f)
```

We use this in `generate_colab.py` so the source of truth is a
human-readable Python file that programmatically produces the notebook.
Edits happen in the .py file, regenerate the notebook with one command:

```bash
python generate_colab.py
```

### 7.2 Subprocess management (server + tunnel)

Pattern for running a long-lived server inside a notebook cell:

```python
import subprocess
import time

# Open log file with line buffering -- critical for live tailing
log_file = open("/tmp/server.log", "w", buffering=1)

# Start server with PYTHONUNBUFFERED to defeat block buffering inside it
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
    env=env,
    stdout=log_file,
    stderr=subprocess.STDOUT,
)

# Poll until /health responds
for _ in range(120):
    try:
        r = urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        if r.status == 200:
            break
    except Exception:
        time.sleep(1)

# ... start cloudflared tunnel similarly ...
```

**Three things to remember:**

1. **`PYTHONUNBUFFERED=1`** in the env, plus `buffering=1` on the
   log file. Without both, your logs come in 4KB bursts.
2. **Poll for readiness** instead of `time.sleep(60)`. Servers warm up
   at different speeds depending on cold cache, weight download,
   model load size.
3. **Don't use `&` in shell** (`!python server.py &`) — Colab doesn't
   reliably propagate signals to backgrounded shell processes. Use
   `subprocess.Popen` from Python.

### 7.3 Killing zombie processes between cell re-runs

Colab's "stop cell" button sends `KeyboardInterrupt` to the cell's
Python, but it doesn't reliably kill subprocesses spawned via Popen.
Result: the *new* server cell tries to bind `:8000` and fails (or
silently uses the *old* server's port).

**Always kill explicitly before restarting:**

```python
!pkill -f "uvicorn app.main:app" 2>/dev/null
!pkill -f cloudflared 2>/dev/null
!sleep 1 && echo "killed"
```

Better: have the cell's `try/finally` do its own cleanup on
KeyboardInterrupt:

```python
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server_proc.terminate()
    tunnel_proc.terminate()
```

### 7.4 The pip + already-imported-PIL problem

Colab pre-imports PIL and numpy in the kernel. If you `pip install`
something that *upgrades* PIL or numpy, you get this:

```
WARNING: The following packages were previously imported in this runtime: [PIL, numpy]
You must restart the runtime in order to use newly installed versions.
```

Mitigation strategies:

1. **Pin to the version Colab already has** — `pip install rembg<2.0.70`
   instead of letting it upgrade to a version that requires newer
   numpy.
2. **Restart runtime after install** — `os._exit(0)` from a cell
   forces a kernel restart. User has to manually re-run subsequent
   cells.
3. **Install before any imports** — first cell `pip install` everything
   you need. Subsequent cells can then `import` freely.

We chose (1) for `rembg`: pinned `rembg>=2.0.50,<2.0.70` so it doesn't
drag in a numpy upgrade.

### 7.5 `nvdiffrast` and `--no-build-isolation`

`nvdiffrast`'s `setup.py` does `import torch.utils.cpp_extension` to
get CUDA build flags. By default, pip creates an isolated build
environment for each wheel-build, and that environment **doesn't have
torch installed**, so the import fails.

```bash
# WRONG -- isolated build env has no torch
pip install git+https://github.com/NVlabs/nvdiffrast/

# RIGHT -- use the main env that has torch
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/
```

Watch for any package whose `setup.py` imports a runtime dep — they
all need `--no-build-isolation`.

### 7.6 Keep retries on `git clone`

Colab VMs hit transient network issues fairly often. A bare:

```python
!git clone https://github.com/foo/bar
```

can fail with a 100-second timeout. Wrap in a retry loop:

```python
for attempt in range(5):
    result = subprocess.run(["git", "clone", url, dest])
    if result.returncode == 0:
        break
    shutil.rmtree(dest, ignore_errors=True)
    time.sleep(2 ** attempt)  # exponential backoff
```

---

## 8. The InstantMesh architecture

Worth understanding what the model actually does, since it informs how
we wire it.

### 8.1 The four-stage pipeline

```
Input image  ─────────────►  6 multi-view images  ───────────►  Triplane features ──────►  3D mesh
   (your photo)                  (zero123plus,                      (DiNo encoder           (FlexiCubes
                                  diffusion model)                   + transformer)          extraction)
```

**Stage 1: Background removal (rembg + U2Net)**
Optional. Crops out the foreground. Without this, the model thinks
the background is part of the object.

**Stage 2: Multi-view generation (zero123plus diffusion)**
A specially-trained diffusion model (sudo-ai/zero123plus-v1.2). Takes
1 image, outputs a fixed grid of 6 views from canonical angles.
Heaviest single step: 75 denoising steps × ~0.5 sec each on T4 = ~38 sec.

**Stage 3: Triplane encoding (DiNo + transformer)**
Encode the 6 views into a triplane representation (3 orthogonal 2D
feature planes that together describe a 3D feature field). Uses a
DiNo ViT (vision transformer) to extract per-view features, then a
transformer to fuse them into the triplane.

**Stage 4: Mesh extraction (FlexiCubes)**
Sample the triplane on a 3D grid (128³ for instant-mesh-large), use
a learned signed distance function and FlexiCubes algorithm to
extract a triangle mesh. This is the memory-hungry step.

### 8.2 Why config matters

`instant-mesh-large` vs `instant-mesh-base`:

| Param | Large | Base |
|---|---|---|
| `triplane_dim` | 80 | 40 |
| `transformer_layers` | 16 | 12 |
| `rendering_samples_per_ray` | 128 | 96 |
| `grid_res` | 128 | 128 |
| Peak VRAM during mesh extraction | ~22 GB | ~10 GB |
| T4 (16 GB) compatible? | NO | yes |
| A100 (40 GB) compatible? | yes | yes |

The biggest knob is `triplane_dim`. Halving it halves the size of the
sampled feature tensor in the mesh extraction step, which is the
allocation that OOMs on T4.

### 8.3 The vendored `pipeline.py` mystery

InstantMesh's GitHub repo includes
`models/InstantMesh/zero123plus/pipeline.py`. Their `run.py` doesn't
use it directly — it does `custom_pipeline="zero123plus"` and relies
on the diffusers community-pipelines mirror. So why is the vendored
file there?

Best guess: the InstantMesh team knew the mirror was fragile and
shipped a known-good copy as a fallback. Modern projects should
**always** use the vendored copy, not the mirror name:

```python
custom_pipeline=str(INSTANTMESH_CODE_DIR / "zero123plus" / "pipeline.py")
```

---

## 9. Operational patterns

### 9.1 The "real vs mock" indicator

Any system that has a fallback mode should explicitly signal which
mode it's running in. Three places to surface this:

**1. Stdout at startup:**
```
[InstantMesh] REAL model loaded successfully
```
or
```
[InstantMesh] Real load FAILED -> falling back to MOCK: <reason>
```

**2. `/health` endpoint:**
```json
{ "status": "healthy", "using_mock": false, "device": "cuda" }
```

**3. Per-job:**
```
[InstantMesh] Real inference FAILED -> falling back to MOCK for this job: <reason>
```

The third is critical: even if the model loaded successfully, an
individual request might OOM and fall back. Per-job logging tells you
*which* requests degraded.

### 9.2 Log latency = real latency

A request that takes 0.8 seconds to complete is doing a mock-mode
icosphere generation. A request that takes 60+ seconds is doing real
ML inference. Use the wallclock as a sanity check:

```python
import time

t0 = time.time()
preview_path = coarse_generator.generate(...)
elapsed = time.time() - t0

if elapsed < 5 and not coarse_generator._using_mock:
    print("[WARN] Inference completed in <5s but using_mock=False -- something is wrong")
```

### 9.3 Pre-flight checks

For long-running ML servers, do a "smoke test" at startup that exercises
every external dep before opening the port. Better to fail fast than
serve broken responses for 8 hours.

```python
def smoke_test():
    problems = []

    # Code on disk
    if not os.path.exists("/path/to/model.ckpt"):
        problems.append("model checkpoint missing")

    # Critical Python imports
    for mod in ["torch", "diffusers", "transformers", "rembg", "nvdiffrast"]:
        try:
            importlib.import_module(mod)
        except Exception as e:
            problems.append(f"{mod}: {e}")

    # GPU available
    if not torch.cuda.is_available():
        problems.append("CUDA not available")

    if problems:
        print("PRE-FLIGHT FAILED:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("PRE-FLIGHT PASSED")
```

We do this in the Colab notebook *before* starting the server. Any
fundamental issue (missing model file, broken dep) is caught in-cell
where it's visible, not deep inside the server's redirected stdout.

### 9.4 Idempotent operations

Operations that can run multiple times without changing the outcome.
Useful for retries, restarts, and reload buttons.

```python
class CoarseGenerator:
    def load(self):
        if self._loaded:
            return  # idempotent: safe to call repeatedly
        with self._load_lock:
            # double-check inside the lock
            if self._loaded:
                return
            self._load_real_model()
            self._loaded = True
```

Pattern: `if (already done): return`, then a lock for thread safety,
then a *re-check* inside the lock (in case another thread did it
while we were waiting).

---

## 10. General software lessons

### 10.1 Read the actual error, not the apparent error

```
Tried to allocate 15.00 GiB. GPU 0 has a total capacity of 14.56 GiB
```

If you skim, you might think "OOM, free some memory." If you read the
numbers, you realize "the request is bigger than the entire GPU." Two
very different problems with very different fixes.

We almost spent time tweaking `empty_cache()` and offloading
strategies before noticing 15 > 14.56. The numbers in error messages
are there for a reason. Read them.

### 10.2 Error messages are evidence, not noise

```
ImportError: cannot import name 'find_pruneable_heads_and_indices' from 'transformers.pytorch_utils'
```

Each part of this is information:
- `ImportError` → it's a Python import problem, not a runtime problem
- `cannot import name X` → the *module exists*, but X isn't in it
- `from transformers.pytorch_utils` → which module
- `find_pruneable_heads_and_indices` → which symbol

Combined: "transformers.pytorch_utils exists, but find_pruneable_heads_and_indices isn't there." That tells you exactly: *this symbol was removed/relocated*. Now you know to either pin transformers or shim the symbol back.

A vague error tells you what failed. A specific error tells you why.

### 10.3 Failure shape tells you what to do

Different failure shapes call for different responses:

| Shape | Likely cause | First thing to try |
|---|---|---|
| `ImportError: No module named X` | X isn't installed | `pip install X` |
| `ImportError: cannot import name X from Y` | X was removed/renamed in Y | Check Y's release notes; pin or shim |
| `AttributeError: 'X' object has no attribute 'foo'` | X's API changed | Same as above |
| `ValueError: <param> must be in <range>` | Wrong arg type or magnitude | Check the function signature |
| `OutOfMemoryError: tried N, total M` | If N > M: config too big. If N < M: free more | Math first, then mitigate |
| `RuntimeError: CUDA error: device-side assert` | Tensor with bad indices/shape on GPU | Re-run on CPU for clearer error |
| `ConnectionError / TimeoutError` | Network flake | Retry with backoff |

### 10.4 Defense in depth

Multiple lines of defense for the same failure mode:

```python
# 1. Pre-flight check (before server starts)
verify_dep("rembg")

# 2. Function-level import (load failures contained)
def _load_real_model(self):
    import rembg
    ...

# 3. Try/except around load (graceful fallback)
try:
    self._load_real_model()
except Exception:
    self._using_mock = True

# 4. Real-vs-mock indicator (visibility)
if self._using_mock:
    log.warning("Falling back to mock")

# 5. Test in CI / smoke test
def test_real_load():
    g = CoarseGenerator()
    g.load()
    assert not g._using_mock
```

Any one of these would have caught the rembg→onnxruntime issue early.
We had only #3, and the fact that it was silent and broad meant the
bug propagated all the way to user-visible icospheres.

### 10.5 "Hard but visible" beats "easy but invisible"

A noisy, in-your-face error message that crashes the program at startup
is **far better** than a silent fallback that ships subtly wrong
output to users.

The original code flipped this:

```python
try:
    self._load_real_model()
except Exception as e:
    print(f"FAILED: {e}")  # prints once, then stdout is buffered
    self._using_mock = True  # silently degrades for the rest of the session
```

The reasoning was probably "we don't want the server to crash if the
GPU isn't available." Fine — but communicate that loudly:

```python
try:
    self._load_real_model()
except Exception:
    if STRICT_MODE:                    # configurable
        raise                          # crash loudly in production
    log.warning("REAL MODEL UNAVAILABLE -- serving MOCK responses", exc_info=True)
    self._using_mock = True
```

When in doubt, fail loudly. Silent degradation is a user trust
problem, not a robustness feature.

---

## Quick reference card

Things to remember at 3am:

```python
# Import verification (don't use find_spec for this)
importlib.import_module("foo")

# Force-flush all output
print("debug", flush=True)

# Line-buffer a log file
open("/tmp/log", "w", buffering=1)

# Always set PYTHONUNBUFFERED for subprocess servers
env["PYTHONUNBUFFERED"] = "1"

# Reduce CUDA fragmentation
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Park a model on CPU to free VRAM (don't run it there)
self.pipeline.to("cpu"); torch.cuda.empty_cache()
self.pipeline.to(self.device)  # before next use

# Don't block the FastAPI event loop on sync work
result = await asyncio.to_thread(blocking_function, *args)

# Serialize GPU access across threads
with self._gpu_lock:
    self._real_inference(...)

# Trimesh: detect if mesh has real colors
mesh.visual.kind in ("vertex", "face")

# Trimesh: don't blindly assign UVs (it nukes ColorVisuals)
if not _has_color_visuals(mesh):
    mesh.visual.uv = ...

# Diffusers: prefer absolute path for custom_pipeline
custom_pipeline=str(local_pipeline_py)  # not bare name

# When transformers removes an internal API
def _shim_compat():
    if not hasattr(SomeClass, "removed_method"):
        SomeClass.removed_method = lambda self, ...: ...
```

---

## See also

- [`CONTEXT.md`](CONTEXT.md) — original project overview and architecture
- [`problem_and_sol.md`](problem_and_sol.md) — chronological bug-by-bug log
- [InstantMesh GitHub](https://github.com/TencentARC/InstantMesh) — upstream model
- [`app/pipeline/instantmesh.py`](app/pipeline/instantmesh.py) — the actual implementation
- [`generate_colab.py`](generate_colab.py) — notebook generator
