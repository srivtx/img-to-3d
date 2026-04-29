# Problem & Solution Log — Wiring the Real InstantMesh Model

This document records the debugging journey from "the app falls back to a mock
icosphere" to "the app generates a real 3D mesh on Colab GPU." Read it
top-to-bottom to understand both *what* was broken and *why* — useful if
you're debugging a similar problem in the future, or just want to remember
how the pieces fit together.

---

## TL;DR

The InstantMesh model was never actually wired. Every job in the queue
quietly fell back to a generated icosphere because the model load was
catching its own exceptions and demoting itself to mock mode. Fixing it
required walking through six independent failures, each one revealed only
after the previous one was resolved:

1. **`onnxruntime` was missing** — `rembg` couldn't even import.
2. **Diffusers couldn't fetch `zero123plus.py`** — the URL it expected (a
   version-pinned community-pipelines mirror) doesn't exist for
   modern Diffusers.
3. **`find_pruneable_heads_and_indices` was removed in transformers 5.x** —
   InstantMesh's vendored DiNo encoder couldn't import.
4. **`get_head_mask` was removed in transformers 5.x** — DiNo couldn't run
   forward pass at inference time.
5. **`instant-mesh-large` was OOM on T4** — one tensor allocation needed
   ~15 GB on a 14.6 GB GPU.
6. **Diffusion pipeline was hogging VRAM** — pipeline + recon model
   couldn't both fit in 16 GB.

After fixing all six, the full pipeline runs end-to-end on a Colab T4 in
~60–90 seconds and produces a real GLB instead of an icosphere.

---

## Initial State (the bug)

The project's CONTEXT.md documented that everything except the model
integration worked. `app/pipeline/instantmesh.py` had a singleton class
called `CoarseGenerator` whose `_load_real_model()` method was wrapped in a
broad `try/except` that, on any exception, would print one short line and
flip `self._using_mock = True`:

```python
try:
    self._load_real_model()
except Exception as e:
    print(f"[InstantMesh] Real load FAILED -> falling back to MOCK: {e}")
    self._using_mock = True
```

Every subsequent `generate()` call short-circuited into the mock icosphere
generator. The exception was being swallowed and not visible in the server
logs because of how Uvicorn's stdout was buffered, so for a long time the
*real* error was invisible.

---

## The Six Failures (in order they were uncovered)

### 1. `ModuleNotFoundError: No module named 'onnxruntime'`

**Root cause.** `rembg` (background removal library) needs `onnxruntime`
to load its ONNX-format U2Net model. In `rembg` ≥2.0.50 the dependency on
`onnxruntime` is an *optional extra* (`rembg[cpu]` or `rembg[gpu]`), not a
hard requirement. `pip install rembg` installs the library but **not**
`onnxruntime`, so `import rembg` blows up at the very first
`from .bg import remove` line in its `__init__.py`.

Why it surfaced as a "model load failed" error: in our
`_load_real_model()` we do `import rembg` to set up a session, and the
import error propagates up.

**Fix.** Add `onnxruntime` to the targeted pip install list in
`generate_colab.py`. The notebook now installs:

```
pytorch-lightning, omegaconf, einops, rembg<2.0.70, onnxruntime,
xatlas, plyfile, PyMCubes, opencv-python, imageio, ninja,
diffusers>=0.27,<1.0, transformers>=4.36
```

**Bonus fix.** The pre-flight smoke test was using
`importlib.util.find_spec(mod)` which only checks if the package's
directory exists on disk — it never executes the module's `__init__.py`.
So `find_spec("rembg")` returned a spec, the smoke test passed, and the
real failure only happened deep in the server's stdout-buffered subprocess.
Switched to `importlib.import_module(mod)` so we now actually run the
import and catch transitive failures (like the missing onnxruntime) loudly,
in-cell, *before* starting the server.

**Commit.** `6b6f2be fix(colab): install onnxruntime + actually-import deps in pre-flight`

---

### 2. Diffusers can't find `zero123plus.py`

**Symptom.**
```
Could not locate the pipeline.py inside zero123plus.
404 Not Found for url 'https://huggingface.co/datasets/diffusers/community-pipelines-mirror/resolve/main/v0.37.1/zero123plus.py'
```

**Root cause.** InstantMesh's official `run.py` and `app.py` call:

```python
DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="zero123plus",        # short name with no slash
    torch_dtype=torch.float16,
)
```

When `custom_pipeline` is a bare name (no slash), Diffusers interprets it
as a community pipeline and looks it up at:

```
https://huggingface.co/datasets/diffusers/community-pipelines-mirror/resolve/main/v{your_diffusers_version}/{name}.py
```

That mirror has many community pipelines but **never had `zero123plus.py`
at any version we tested** (v0.27 through v0.36 all 404). InstantMesh was
written when zero123plus.py lived in a different place; the mirror's path
structure changed and zero123plus was lost.

**The good news.** The InstantMesh repository itself ships a vendored copy
of `pipeline.py` at `models/InstantMesh/zero123plus/pipeline.py` — exactly
because the upstream lookup was already known to be fragile. We just need
to point Diffusers at that local file.

**Fix.** In `_load_real_model()`, build an absolute path to the vendored
file and pass it as `custom_pipeline=str(local_path)`:

```python
local_pipeline_py = INSTANTMESH_CODE_DIR / "zero123plus" / "pipeline.py"
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline=str(local_pipeline_py),
    torch_dtype=torch_dtype,
)
```

Diffusers happily loads the class from disk. No network, no version-mirror
dependency, and the file is the one the InstantMesh team verified against
their model weights.

**Commit.** `9ffc071 fix(instantmesh): use vendored zero123plus pipeline.py instead of community mirror`

---

### 3. `transformers 5.x` removed `find_pruneable_heads_and_indices`

**Symptom.**
```
ImportError: cannot import name 'find_pruneable_heads_and_indices'
            from 'transformers.pytorch_utils'
```

**Root cause.** InstantMesh's vendored DiNo encoder
(`models/InstantMesh/src/models/encoder/dino.py`) does:

```python
from transformers.pytorch_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
```

These were public-but-internal helpers in transformers 4.x. transformers
5.x cleaned up `pytorch_utils` and removed/relocated them. Modern Colab
installs transformers 5.x by default, so the import blows up the first
time `instantiate_from_config(config.model_config)` triggers DiNo's module
load.

**Why we didn't just downgrade transformers.** Forcing
`transformers<5.0` would mean a multi-hundred-MB pip operation that
risks dependency conflicts with `diffusers`, `accelerate`, and
`huggingface-hub`, all of which pin minimum transformers versions. Slow
and fragile.

**Fix.** Add a small in-process shim that re-injects the missing helpers
into `transformers.pytorch_utils` *before* InstantMesh's encoder imports
them. The implementations are tiny pure-PyTorch functions (10-20 lines
each) lifted verbatim from transformers v4.46.0:

```python
def _shim_transformers_compat() -> None:
    import transformers.pytorch_utils as tpu
    if not hasattr(tpu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(...):
            ...
        tpu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    # ... same pattern for prune_linear_layer ...
```

Called from `_load_real_model()` right after we add InstantMesh's `src/`
to `sys.path`, but before any InstantMesh code path imports transformers.

**Commit.** `5f91158 fix(instantmesh): shim removed transformers.pytorch_utils helpers for InstantMesh DiNo encoder`

---

### 4. `transformers 5.x` removed `PreTrainedModel.get_head_mask`

**Symptom.** Discovered only after #3 was fixed and the model successfully
loaded *and* ran 75 zero123plus diffusion steps:

```
File ".../dino.py", line 503, in forward
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
AttributeError: 'ViTModel' object has no attribute 'get_head_mask'
```

**Root cause.** `get_head_mask` was a method on `ModuleUtilsMixin` (mixed
into `PreTrainedModel`) in transformers ≤4.x. Same internal cleanup that
deleted the helpers in #3 also removed this method. InstantMesh's
vendored ViTModel calls `self.get_head_mask(...)` inside `.forward()`,
which now blows up at the first inference call (after 50 seconds of
useful diffusion work — very satisfying when this is the only thing
between you and a real mesh).

**Fix.** Extend `_shim_transformers_compat()` to also monkey-patch
`get_head_mask` (and its tiny helper `_convert_head_mask_to_5d`) onto
`transformers.PreTrainedModel`. Both methods are short, pure-PyTorch,
and lifted verbatim from transformers v4.46.0. Monkey-patching at the
class level means *all* existing `ViTModel` instances pick it up
automatically (Python looks up methods via the class, not the instance).

We also confirmed by grepping the entire `dino.py` and `dino_wrapper.py`
that `get_head_mask` is the **only** other base-class method either of
them calls (besides `post_init`, which still exists in transformers 5.x).
No more shims of this family expected.

**Commit.** `5842ef8 fix(instantmesh): also shim PreTrainedModel.get_head_mask for inference`

---

### 5. T4 OOM at `instant-mesh-large` config

**Symptom.** After fixes #1–4, inference advanced through diffusion and
the DiNo encoder, hit `model.extract_mesh()`, and OOMed:

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 15.00 GiB.
GPU 0 has a total capacity of 14.56 GiB of which 6.91 GiB is free.
```

Read that carefully: **the operation wants to allocate one tensor that's
larger than the entire GPU**. Even on a freshly-rebooted T4 with zero
other processes, this single allocation can never succeed. Not a code
bug — a config-vs-hardware mismatch.

**Root cause.** InstantMesh's flagship config `instant-mesh-large` was
designed for A100/V100 GPUs (40–80 GB VRAM):

```yaml
triplane_dim: 80          # half this on -base
transformer_layers: 16    # 12 on -base
grid_res: 128
```

At FlexiCubes mesh extraction, `synthesizer_mesh.py` does:

```python
grid_features = torch.index_select(
    input=sampled_features,
    index=flexicubes_indices.reshape(-1),
    dim=1,
)
```

For grid_res=128, `flexicubes_indices.reshape(-1)` is ~16M elements; the
output shape is `[1, 16M, F]` where F scales with `triplane_dim`. With
`triplane_dim=80`, F ends up around 256 after the MLP, producing
16M × 256 × 4 bytes ≈ 16 GB. T4 cannot fit this regardless of what else
is loaded.

**Fix.** Switch the default config from `instant-mesh-large` to
`instant-mesh-base`, which uses `triplane_dim=40` (half) and
12 transformer layers (vs 16). The 16M tensor halves to ~8 GB, which
fits. Mesh quality drops slightly but stays good.

The notebook now sets:

```python
env["INSTANTMESH_CONFIG"] = "instant-mesh-base"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

Users on bigger GPUs (A100/V100/RTX 3090+) can flip back to
`instant-mesh-large` by editing one line.

**Commit (combined with #6).** `ba94bcb fix(instantmesh): T4-friendly config + offload diffusion pipeline between stages`

---

### 6. Pipeline + recon model can't both fit on T4

**Root cause.** Even on `instant-mesh-base`, with the diffusion
pipeline (~3 GB) and the reconstruction model (~3 GB) both resident on
the T4, the FlexiCubes mesh extraction step can still hit the ceiling
during a peak allocation. It works on a clean GPU but fails after a
warm-up.

**Fix.** Offload the diffusion pipeline to CPU between the two stages
of each request. We don't need it again until the *next* request's view
generation, and it's sitting on ~3 GB of VRAM that the FlexiCubes step
desperately wants:

```python
# After zero123plus generates the 6 views:
if self.device.type == "cuda":
    try:
        self.pipeline.to("cpu")
    except Exception:
        pass
    torch.cuda.empty_cache()

# At the top of the next _real_generate, before diffusion runs:
if self.device.type == "cuda":
    try:
        self.pipeline.to(self.device)
    except Exception:
        pass
```

Round-trip cost: ~1–2 seconds. Negligible against the 50-second
diffusion step.

**Commit (same as #5).** `ba94bcb`

---

## Final State

After all six fixes, on a Colab T4:

| Stage | Time |
|---|---|
| First-time setup (clone + install + 7 GB weights download) | ~5 min |
| Server cold start (model load into VRAM) | ~60 s |
| Per-request diffusion (75 zero123plus steps) | ~50 s |
| Per-request triplane + mesh extraction | ~10–15 s |
| **Total per generation (after warm-up)** | **~60–90 s** |

The server hands out a cloudflare tunnel URL; uploads through the web UI
hit `/generate-3d`, get queued, run real InstantMesh inference, and
return a real `final.glb` that renders in the Three.js viewer.

---

## Files Changed (just the meaningful diffs)

```
app/pipeline/instantmesh.py
  + _shim_transformers_compat() (~80 lines)
  + Vendored zero123plus pipeline.py path resolution
  + Pipeline offload-to-CPU between request stages
  + Lazy thread-safe model loading singleton

generate_colab.py
  + onnxruntime in pip install list
  + import_module-based pre-flight smoke test
  + INSTANTMESH_CONFIG default = instant-mesh-base
  + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

colab/Image_to_3D_Generator.ipynb
  (regenerated from generate_colab.py)
```

---

## What Each Failure Taught Us

1. **Don't trust `find_spec` to mean "this package works"** — it only
   checks the directory exists. Always actually import.
2. **Pre-flight every external assumption.** Community pipeline mirrors,
   transformers internal APIs, and GPU memory limits all changed under
   InstantMesh's feet between the time it was written and now.
3. **Buffered stdout hides real errors.** If a subprocess looks like it
   "just decided" to fall back to mock mode, it's because the actual
   exception is sitting in a buffer somewhere. Print explicitly to
   `sys.stderr` and/or set `PYTHONUNBUFFERED=1`, or test the load path
   in a foreground cell to bypass buffering entirely.
4. **Monkey-patching is the right call when downgrading is risky.** The
   alternative — pinning transformers to <5.0 — would have triggered a
   200 MB pip dance and risked four other deps breaking. A 30-line shim
   was strictly better.
5. **Hardware mismatch looks like a code bug.** "Tried to allocate
   15.00 GiB on a 14.56 GiB GPU" is a math problem, not a Python problem.
   When the requested size exceeds total VRAM, no amount of
   `empty_cache()` will help — change the config.

---

## Re-running the Whole Thing

```
# Open in Colab from GitHub (always gets the latest):
https://colab.research.google.com/github/srivtx/img-to-3d/blob/main/colab/Image_to_3D_Generator.ipynb

# Run cells top to bottom. The pre-flight cell (Step 4) will fail loudly
# if any of the six issues come back. The server cell (Step 5) will print
# the cloudflare URL when ready.
```

If something new breaks, paste the traceback and treat it as a fresh
domino — the pattern of each failure being one layer deeper than the
last has held throughout this debugging session.
