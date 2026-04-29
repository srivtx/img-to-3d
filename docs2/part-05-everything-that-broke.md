# Part 5: Everything That Broke — A Chronicle of Errors

## The Domino Pattern

We didn't fix one bug and suddenly everything worked. We fixed **seven bugs in a row**, each revealing the next one.

This is the "deepest visible failure" pattern. When you fix a surface bug, you don't see success — you see the **next bug that was hiding behind it**.

**Rule:** If the error moves to a different file, you're making progress.

---

## Error 0: The Silent Fallback

**Symptom:** Users upload photos. They get a GLB file. They see a smooth ball in the 3D viewer. It looks "fine."

**The truth:** The real AI model never loaded. The app silently fell back to a mock icosphere.

**Why this was confusing:** The API returned HTTP 200. The file downloaded. The viewer rendered it. Everything looked like success. But it was a sphere.

**Root cause:**
```python
try:
    load_real_model()
except Exception:
    # Oops — we caught EVERY error and hid it
    use_mock_mode()
```

**The `except Exception:` swallowed:**
- Missing dependencies
- Import errors
- CUDA out of memory
- Wrong file paths
- Version mismatches

**Lesson:** Never catch `Exception` without logging the full traceback. If you do, users think your app works when it doesn't.

**Fix:**
```python
try:
    load_real_model()
except Exception:
    traceback.print_exc()  # SHOW THE ERROR
    use_mock_mode()
```

---

## Error 1: Buffered Logs Hid the Real Error

**Symptom:** Server logs showed old messages. New errors appeared minutes late or never.

**Why:** Python's `print()` is **block-buffered** when stdout goes to a pipe/file instead of a terminal.

```python
server_proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "app.main:app"],
    stdout=subprocess.PIPE,  # Redirected → buffered!
)
```

Python collects output in a 4 KB buffer. It only flushes when:
- The buffer fills up
- The process exits
- You explicitly call `flush=True`

**Result:** The error "Real load FAILED" was sitting in the buffer, invisible.

**Fix:** Line-buffer the output file:
```python
log_file = open("/tmp/server.log", "w", buffering=1)  # Line-buffered
```

Or set the environment variable:
```python
env["PYTHONUNBUFFERED"] = "1"
```

Or use `flush=True` on critical prints:
```python
print("[ERROR] Model failed to load", flush=True)
```

**Lesson:** If a subprocess seems to "decide" things silently, suspect buffered output.

---

## Error 2: `onnxruntime` Missing

**Symptom:** `ModuleNotFoundError: No module named 'onnxruntime'`

**Where it happened:** Inside `import rembg`

**Why:** `rembg` (background removal) depends on `onnxruntime` (ONNX model runtime). But `onnxruntime` is an **optional dependency** of rembg:

```toml
# rembg's pyproject.toml
[project.optional-dependencies]
cpu = ["onnxruntime"]
gpu = ["onnxruntime-gpu"]
```

So `pip install rembg` installs the Python code but NOT the runtime.

**The trap:** `pip install rembg` says "success." But `import rembg` crashes.

**Fix:** Explicitly install the runtime:
```bash
pip install rembg onnxruntime
```

**Lesson:** After `pip install`, always verify with:
```bash
python -c "import the_package; print('OK')"
```

---

## Error 3: `find_spec` vs `import_module`

**Symptom:** Pre-flight check said "rembg is installed" but import still failed.

**Why:** We used `importlib.util.find_spec("rembg")` which only checks if a folder exists. It does NOT run the package's `__init__.py`.

```python
# WRONG — only checks folder existence
if importlib.util.find_spec("rembg") is not None:
    print("rembg is OK")  # Lies! It might still crash on import.

# RIGHT — actually imports and runs the code
try:
    importlib.import_module("rembg")
    print("rembg is OK")
except Exception as e:
    print(f"rembg is BROKEN: {e}")
```

**Lesson:** `find_spec` is for "can Python find this?" `import_module` is for "will this actually run?"

---

## Error 4: Diffusers Community Pipeline 404

**Symptom:** HTTP 404 when loading the Zero123++ pipeline:
```
https://huggingface.co/datasets/diffusers/community-pipelines-mirror/...
```

**Why:** Diffusers' `custom_pipeline` argument has three modes:

```python
# Mode 1: Bare name → looks in community mirror (FRAGILE)
custom_pipeline="zero123plus"
# Tries: .../v0.37.1/zero123plus.py → 404!

# Mode 2: Repo ID → looks in HF Hub repo
custom_pipeline="sudo-ai/zero123plus-v1.2"

# Mode 3: Absolute path → loads file directly (RELIABLE)
custom_pipeline="/content/img-to-3d/models/InstantMesh/zero123plus/pipeline.py"
```

The community mirror doesn't have every pipeline at every Diffusers version.

**Fix:** Point directly at the vendored pipeline file:
```python
local_pipeline = INSTANTMESH_CODE_DIR / "zero123plus" / "pipeline.py"
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline=str(local_pipeline),
)
```

**Lesson:** For production, always use absolute paths to vendored code. Don't rely on network lookups.

---

## Error 5: Transformers 5.x Removed Internal Helpers

**Symptom:**
```
ModuleNotFoundError: No module named 'transformers.pytorch_utils'
AttributeError: 'ViTModel' object has no attribute 'get_head_mask'
```

**Why:** InstantMesh's vendored DiNo encoder imports from `transformers.pytorch_utils`:
```python
from transformers.pytorch_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
```

These were **internal helpers** in transformers ≤4.x. In transformers 5.x, they were removed as part of cleanup.

**Two options:**

**Option A: Pin to old version**
```
transformers<5.0
```
Problem: Forces a 200 MB re-install, might break diffusers/accelerate.

**Option B: Shim (re-implement the missing functions)**
```python
def _shim_transformers_compat():
    import transformers.pytorch_utils as tpu
    from transformers import PreTrainedModel
    
    if not hasattr(tpu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(...):
            # Re-implemented from transformers v4.46.0
            ...
        tpu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    
    if not hasattr(PreTrainedModel, "get_head_mask"):
        def get_head_mask(self, ...):
            # Re-implemented from transformers v4.46.0
            ...
        PreTrainedModel.get_head_mask = get_head_mask
```

**We chose Option B** because:
- The helpers are small (15 lines each)
- Pure PyTorch, no version-specific behavior
- Avoids dependency hell

**Lesson:** Internal APIs can disappear. Either pin versions or shim. For small, pure functions: shim.

---

## Error 6: CUDA Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

**Why:** T4 GPU has 16 GB VRAM. Peak usage:
- Diffusion pipeline: ~3 GB
- Reconstruction model: ~2 GB
- Intermediate tensors: ~6 GB
- Fragmentation overhead: ~2 GB
- **Total:** ~13 GB

**The trigger:** Loading both models simultaneously + running diffusion with large batches.

**Fix 1: Offload pipeline to CPU between requests**
```python
# After diffusion finishes
pipeline.to("cpu")
torch.cuda.empty_cache()

# Before next diffusion request
pipeline.to("cuda")
```

This frees ~3 GB. Cost: 1-2 seconds to move back to GPU.

**Fix 2: Use smaller model**
```
instant-mesh-base  (instead of instant-mesh-large)
```
Base model uses ~30% less VRAM.

**Fix 3: Enable expandable segments**
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

This reduces memory fragmentation.

**Lesson:** GPU memory is finite. Track peak usage. Offload when possible.

---

## Error 7: Colab Working Directory Bug

**Symptom:**
```
shell-init: error retrieving current directory: getcwd: cannot access parent directories
```

**Why:** The notebook did:
```python
import shutil
shutil.rmtree("/content/img-to-3d")  # Delete the folder we're IN
```

After deletion, the shell's current working directory is a deleted folder. Every subsequent command fails.

**Fix:** Change directory BEFORE deleting:
```python
import os
os.chdir("/content")  # Move OUT first
shutil.rmtree("/content/img-to-3d")  # Now safe to delete
```

**Lesson:** Never delete the folder you're standing in.

---

## Error 8: Notebook JSON Corruption

**Symptom:** `SyntaxError: JSON.parse: bad escaped character`

**Why:** We edited `.ipynb` files manually. Inside the JSON, Python strings contain `
` (newline escape). When we edited the file with text editors, some tools converted `
` to actual newlines, breaking the JSON string.

**Example of broken JSON:**
```json
"source": [
    "print(\"hello\n",        ← 
 became a real newline!
    "world\")"
]
```

**Fix:** Generate notebooks programmatically with `nbformat`:
```python
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

nb = new_notebook()
nb.cells = [new_code_cell("print('hello')")]
nbformat.write(nb, "notebook.ipynb")
```

**Lesson:** Never edit `.ipynb` JSON manually. Always use `nbformat` or Jupyter's UI.

---

## Error 9: `nvdiffrast` Compilation Failed

**Symptom:** `nvdiffrast` failed to install because it needs CUDA headers at compile time.

**Why:** `nvdiffrast` is a CUDA-accelerated rasterizer. It contains C++/CUDA code that must be compiled against your specific PyTorch and CUDA versions.

**Fix:** Install with `--no-build-isolation` so it can see PyTorch's CUDA headers:
```bash
pip install nvdiffrast --no-build-isolation
```

**Alternative:** Skip it if not needed. InstantMesh works without nvdiffrast for inference (it's mainly for training).

---

## Error 10: Vertex Colors Lost in Refinement

**Symptom:** The final GLB file lost its colors. Looked gray instead of textured.

**Why:** Trimesh's `export()` creates a new visual object. If the original had vertex colors but the export path creates a texture-based visual, colors are lost.

**Fix:** Preserve vertex colors explicitly:
```python
def process_mesh(input_path, output_path):
    mesh = trimesh.load(input_path)
    
    # Save original colors
    if hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors
    
    # Process...
    mesh = subdivide(mesh)
    mesh = smooth(mesh)
    
    # Restore colors
    if colors is not None:
        mesh.visual.vertex_colors = colors
    
    mesh.export(output_path)
```

**Lesson:** 3D file formats have different ways of storing color (vertex colors, textures, materials). Converting between them can lose data.

---

## Error 11: Subprocess Started in Wrong Directory

**Symptom:** `FileNotFoundError: ./cloudflared not found`

**Why:** The notebook tried to run `./cloudflared` but the current directory wasn't `/content/img-to-3d` anymore (because of the working directory bug).

**Fix:** Always use absolute paths:
```python
subprocess.Popen(
    ["/content/img-to-3d/cloudflared", "tunnel", ...],
    cwd="/content/img-to-3d",
)
```

---

## Error 12: CORS Blocked Local Development

**Symptom:** Browser console shows `CORS policy: No 'Access-Control-Allow-Origin' header`

**Why:** Browser security prevents web pages from making requests to different domains.

**Fix:** Add CORS middleware:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all (development only!)
)
```

---

## Summary: The Fix Timeline

| Order | Error | Fix |
|-------|-------|-----|
| 1 | Silent fallback | Log full traceback |
| 2 | Buffered logs | `PYTHONUNBUFFERED=1` |
| 3 | Missing onnxruntime | Explicit install |
| 4 | `find_spec` lying | Use `import_module` |
| 5 | Pipeline 404 | Use absolute path |
| 6 | Transformers 5.x | Shim removed APIs |
| 7 | CUDA OOM | Offload to CPU |
| 8 | Working dir deleted | `os.chdir` first |
| 9 | JSON corruption | Use `nbformat` |
| 10 | nvdiffrast compile | `--no-build-isolation` |
| 11 | Colors lost | Preserve vertex colors |
| 12 | cloudflared not found | Absolute paths |
| 13 | CORS blocked | Add middleware |

---

## Meta-Lessons

### Lesson 1: Fail Loudly
If something breaks, scream. Don't whisper.

### Lesson 2: Log Everything
You can't debug what you can't see.

### Lesson 3: Isolate Variables
Test the model alone before wiring it to the server.

### Lesson 4: Version Drift is Real
Libraries change. Internal APIs disappear. Pin or shim.

### Lesson 5: GPU Memory is a Budget
Track every MB. Offload when possible.

### Lesson 6: The Web is a Different World
Browser security, CORS, buffering — none of these exist in Python scripts.

---

**Next:** [Part 6: 20+ Alternative Architectures](part-06-alternative-architectures.md) — How else could we have built this? From simple to advanced.
