# Part 05 — ML concepts, one ladder at a time

**Prerequisites:** [Part 02](part-02-from-photo-to-glb-simplest-story.md) (you already know the story of “photo → 3D file”). Here we deepen **slowly**.

---

## Layer A — What is machine learning?

- **Traditional program:** you write **exact rules** (if pixel > 200 then …).
- **Machine learning:** you show **many examples**; the model **learns patterns** from data.

A **neural network** is a big **graph of math** (layers) with **millions of numbers** (**weights**) adjusted during **training**.

---

## Layer B — Supervised learning (the common case)

**Training:** pairs of **input → desired output** (images + labels, etc.).

**Inference** (what we do at deploy time): **new** input → model **predicts** output using **fixed** weights.

Our photogrammetry stack is **inference only** — we **don’t** train InstantMesh at runtime.

---

## Layer C — Convolutions (very short)

**Convolutional layers** slide small filters over an image to detect edges, textures, objects at different scales.

You don’t need the math for day-one debugging — just know: **CNNs** are the standard **image** backbone.

---

## Layer D — From one view to many (zero-shot / multi-view)

A single photo is **ambiguous** — the back isn’t observed.

**InstantMesh** relies on pipelines that synthesize **multiple consistent views** of the object (**novel view synthesis**) so geometry can fuse into a mesh.

Rough mental model:

1. Understand the object somewhat from RGB.
2. **Imagine/hallucinate** other angles (not arbitrarily — learned from lots of paired data conceptually).
3. **Fusion** predicts **signed distance** or occupancy in **3D voxel space**.
4. **Marching cubes** extracts a **surface**.

---

## Layer E — What is diffusion?

**Diffusion models** gradually **denoise** random noise into a structured image/video.

**Why diffusion for views?**

- Extremely high-quality **generation** conditioned on cues (often **CLIP** embeddings from the source image).

In our Diffusers pipelines, **`num_inference_steps`** (e.g. 75) trades **speed vs quality.**

---

## Layer F — What is Transformer / ViT / DiT vocabulary?

**Transformer architecture** scales well with attention.

**ViT (Vision Transformer)** treats image **patches** like tokens — great for semantics.

InstantMesh internals reference **ViT-backed** checkpoints and **attention** patterns.

Libraries like **PyTorch Lightning** historically exposed utilities for **attention head pruning.** Newer **`transformers` 5.x** removed some helpers — downstream code importing old **`pytorch_pretrained_vit`** then **explodes** unless we shim (see Part 09).

---

## Layer G — What is ONNX Runtime here?

Many projects use **`rembg`** for background segmentation (not always required for InstantMesh but present in tooling).

Under the hood **`rembg` → `onnxruntime`** for fast inference graphs.

So **`import rembg`** can fail if ONNX CPU runtime isn’t installed — even if you “pip installed rembg successfully.” Our fix: **`onnxruntime` explicit** in installs + smoke **`import_module`**.

---

## Layer H — Checkpoint vs code

Even perfect code fails if **`models/InstantMesh/zero123plusplus/` weights** aren't present (~4 GB downloads).

Treat **checkpoint + CUDA memory** as coequal prerequisites with syntax.

---

**Next:** [Part 06 — InstantMesh pipeline stages inside this repo](part-06-instantmesh-pipeline-stages.md).
