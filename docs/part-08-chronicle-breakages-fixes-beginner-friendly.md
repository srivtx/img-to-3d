# Part 08 — Chronicle: what broke, why, and how we fixed it (beginner walkthrough)

**Prerequisites:** [Parts 02–07](README.md). This part retells `problem_and_sol.md` **and** distills patterns from `learn.md`, in order — **one failure unlocks the next** (“domino debugging”).

---

## Step 0 — The bug in one sentence

Users thought they were running **InstantMesh**, but the app often showed a **smooth ball** (icosphere). The real neural network **never successfully loaded or ran**, yet the API still returned **HTTP 200** and a **valid GLB**.

**Why that was confusing:** from the outside, “I got a mesh file” looks like success.

---

## Step 1 — Learn the words: “fallback” and “swallowed exception”

Imagine a function whose job is **“load expensive AI”**:

1. Try to load.
2. On **any** error, **print one line** (maybe) and set a flag **`using_mock = True`**.
3. Next time someone asks for 3D, **skip AI** and draw a **sphere** instead.

That is a **fallback**: the program **keeps going** with a **substitute**.

An exception is **swallowed** when **`except Exception:`** catches it — the **crash is hidden** from the caller. You only see clues if **logs** show the real traceback.

**Beginner takeaway:** if output looks “too simple,” **distrust success** until logs say the real model ran.

---

## Step 2 — Why the *first* error stayed invisible (buffering)

Servers often **buffer** `print()` output (hold text in memory, flush later). In Colab/subprocess setups, you might see **`[InstantMesh] FAILED`** **late** or **not at all**.

**Fix pattern (from `learn.md`):**

- **`PYTHONUNBUFFERED=1`**, or **`print(..., flush=True)`**, or line-buffered log files.
- **Pre-flight**: run **`import rembg`** **in a notebook cell** *before* starting Uvicorn so the traceback appears **in front of you**.

---

## Step 3 — Failure 1: missing `onnxruntime`

### What you would see (eventually)

`ModuleNotFoundError: No module named 'onnxruntime'`

### Super-beginner “why”

Many Python packages **declare optional dependencies**.

- **`pip install rembg`** gives you the **Python package**.
- **`rembg`** internally wants **`onnxruntime`** to run a **U2Net** ONNX file.
- Without it, **`import rembg`** **fails** deep inside `rembg`’s **`__init__.py`**.

So the **first line** of failure is not “InstantMesh bad” — it’s **“dependency of a dependency.”**

### Fix

- Add **`onnxruntime`** explicitly to install lists (`generate_colab.py`, `requirements.txt` as applicable).

### Deeper lesson: `find_spec` vs `import_module`

- **`importlib.util.find_spec("rembg")`** only checks whether a **folder** for `rembg` **exists**.
- It **does not run** `rembg/__init__.py`, so it **misses** the missing **`onnxruntime`**.

**Smoke test must use** **`importlib.import_module("rembg")`** (actually import) so transitive errors surface **before** the server starts.

---

## Step 4 — Failure 2: Diffusers could not download `zero123plus.py`

### Symptom

HTTP **404** from a URL like:

`.../community-pipelines-mirror/.../zero123plus.py`

### Beginner explanation

**Diffusers** can load a **custom pipeline** three ways (simplified):

1. **Short name** `"zero123plus"` → tries a **versioned community mirror** (**fragile** — mirrors change).
2. **Repo-style** id → looks in a Hub repo.
3. **Absolute path to `.py`** on **your disk** → **loads that file directly** (**most reliable**).

InstantMesh **ships** its own **`models/InstantMesh/zero123plus/pipeline.py`** because upstream mirrors already broke historically.

### Fix

Pass **`custom_pipeline=str(absolute_path_to_pipeline.py)`** so Diffusers **never hits** the broken mirror.

---

## Step 5 — Failure 3 & 4 — `transformers` 5.x removed internal helpers

Libraries **evolve**. Research code (**vendored DiNo**) did:

```python
from transformers.pytorch_utils import find_pruneable_heads_and_indices, ...
```

Those names **vanished** in **transformers 5.x**.

Later, **`ViTModel.forward`** called **`self.get_head_mask`** — another **removed** mixin method.

### Why not “just downgrade transformers”?

Pinning **`transformers<5`** might **fight** **`diffusers`**, **`accelerate`**, **`huggingface-hub`**, which expect **newer** versions — **dependency hell** + huge reinstalls.

### Fix strategy: shim (tiny compatibility layer)

Before InstantMesh imports those modules:

1. **Define** small pure-PyTorch implementations (copied from an older transformers release).
2. **Attach** them to **`transformers.pytorch_utils`** if missing (`find_pruneable_heads_*`, **`prune_linear_layer`**).
3. **Monkey-patch** **`PreTrainedModel.get_head_mask`** (+ **`_convert_head_mask_to_5d`**) onto the **class** so **all existing instances** get the method.

**Incremental understanding:**

- First fix **import-time** breakage (helpers).
- Next fix **runtime** breakage after heavy diffusion (**get_head_mask**) — **deeper domino.**

---

## Step 6 — Failure 5 — “OOM”: out of GPU memory (math, not morality)

### Symptom

`torch.OutOfMemoryError` — tried to allocate **~15 GiB** on a **~14.56 GiB** T4.

### Beginner intuition

VRAM is **finite**. If **one tensor** needs **more bytes than the GPU has**, **no clever Python** fixes it — you must **change the model config** or **hardware**.

**InstantMesh** has **`instant-mesh-large`** tuned for **40–80 GB** GPUs. On a cheap T4, **grid resolution × feature width** blows up tensors during FlexiCubes extraction.

### Fix

- Default to **`instant-mesh-base`** (smaller **`triplane_dim`**, fewer layers) — halves peak tensor cost (see VRAM sketches in **`learn.md` §4**).
- Set **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** to reduce **fragmentation** (helps when you’re *near* the limit, not when allocation > total VRAM).

---

## Step 7 — Failure 6 — two giants on one small playground

Even after shrinking the recon model config, **`zero123plusplus` diffusion (~GB VRAM)** and **reconstruction/FlexiCubes peak** might **alternate badly**.

### Idea: “Park on CPU”

After views are generated, **`pipeline.to("cpu")`** and **`torch.cuda.empty_cache()`** — **free VRAM** for the next phase. Before the **next request**’s diffusion, move pipeline **back** to **`cuda`** — ~1–2 s overhead vs ~50 s diffusion.

**Warnings** about **`float16`** on CPU may appear; often **noise** vs hard failure (**`learn.md` §4.5**).

---

## Step 8 — Mesh refinement pitfall (not one of the “six”, but equally painful)

Fast simplification **`fast-simplification`** path + wrong **`simplify_quadric_decimation`** call → subtle bugs.

Using **positional** args where Trimesh expects **`face_count=`** accidentally passed **percent semantics** errors.

Operations like **subdivide + UV replacement** dropped **vertex colors** → **`final.glb`** looked **flat white**.

**Fix:** **`mesh_processor.py`** chooses **color-safe** path (**Taubin**, **`simplify_quadric_decimation(face_count=...)`**) vs **texture** path.

---

## Step 9 — Order of fixes (why sequence matters)

| Order | Visible symptom | Category |
|:---:|:---|:---|
| 1 | Import error | Dependencies |
| 2 | 404 / pipeline load | Networking / HF mirrors |
| 3 | ImportError transformers | Library API drift |
| 4 | `get_head_mask` AttributeError | Library API drift (runtime) |
| 5 | CUDA OOM at mesh extract | Hardware vs config |
| 6 | OOM intermittently after partial success | Peak VRAM scheduling |

Fixing (#1) reveals (#2); fixing (#2–4) reaches real inference long enough to hit (#5–6).

---

## Step 10 — What “success” looks like end-to-end (rough Colab T4 timings)

See **`problem_and_sol.md` “Final State”** table — first compile/download slow; **steady-state** requests ~**60–90 s**.

---

## Step 11 — Carry these habits forward

Summarized from **`learn.md`** and this chronicle:

1. **Seek the deepest visible failure** — don’t stop at “mock sphere.”
2. **Smoke test with real imports**, not **`find_spec` alone**.
3. **Prefer vendored/local paths** over **version-pinned URLs**.
4. **Shim narrowly** instead of pinning **half the Python world**.
5. **Measure VRAM** as arithmetic when OOM persists.
6. **Question silent fallbacks** — log loudly, optionally expose **`using_mock`** on **`/health`** (pattern in `learn.md`; wire if you need it).

---

**Next:** cross-cutting techniques → [Part 09](part-09-cross-cutting-async-vram-deps-refinement.md).
