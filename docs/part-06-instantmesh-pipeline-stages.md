# Part 06 — InstantMesh pipeline stages (what happens on GPU)

**Prerequisites:** [Part 05](part-05-ml-concepts-layer-by-layer.md).

---

## Layer A — Big picture

**InstantMesh** = research stack + pretrained weights turning **one RGB photo** into a textured **triangle mesh.**

Layout in many forks matches **`InstantMesh/scripts/instant-mesh-*.py`** style entrypoints — our **`app/pipeline/instantmesh.py`** wraps equivalent stages for production and expects checkpoints under **`models/InstantMesh`**.

Rough pipeline (conceptual):

1. **Load image** on CPU, normalize.
2. **Multiview diffusion** (`zero123++` family): synthesize **novel RGB views** consistent with the photo.
3. **Reconstruction**: fuse views into **an implicit / grid representation** suitable for extracting a surface (architecture-specific).
4. **Surface extraction & baking**: emit **triangle mesh + materials** (often **FlexiCubes-class** extraction — VRAM-heavy at high **`grid_res`**).


---

## Layer B — Why `zero123plus` is special

**Zero123 / Zero123++** class models map **one image + optional pose hint** → **another viewpoint’s RGB.**

If that community pipeline path is **missing** or **404**, nothing else runs — we load **local** `models/InstantMesh/zero123plus/pipeline.py` via Diffusers **`custom_pipeline`**.

---

## Layer C — VRAM phases (why we “park” on CPU)

Each stage grabs **big tensors**.

On **T4 (~15 GB)**, large models can **OOM** if everything stays on GPU simultaneously.

**Mitigation pattern:** after heavy diffusion, **move pipeline components to CPU** **before** marching-cubes style stages (you may see PyTorch warnings about fp16 on CPU — often **benign** for our flow).

Env knob: **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** reduces fragmentation.

Default **base** model instead of **large** when VRAM is tight.

---

## Layer D — Output artifacts

- Intermediate **`.ply`** (point cloud / mesh-ish) may exist for debugging.
- Final **GLB** embeds **geometry + material + texture** for web loading.

---

## Layer E — Where refinement fits

After InstantMesh returns a mesh, **optional** **Trimesh** passes (smooth, decimate) run in `app/services/mesh_processor.py` — **not** part of the neural forward pass but still **CPU/GPU sensitive** for big meshes.

---

**Next:** [Part 07 — Code map: which file does what](part-07-code-map-which-file-does-what.md).
