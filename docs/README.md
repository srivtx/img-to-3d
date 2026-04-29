# Documentation — Beginner‑Oriented Deep Dive

This folder explains **image‑to‑3d** (this repo) the way we wish we had
read it on day zero: **one layer at a time**. Each document adds **one
concept** before the next. If a term confuses you, check **GLOSSARY.md**
first.

## How to read this (recommended order)

| Order | Document | What you will understand after reading |
|:-----:|----------|----------------------------------------|
| 1 | [`part-01-reading-order-and-philosophy.md`](part-01-reading-order-and-philosophy.md) | Why these docs exist and how complexity is layered. |
| 2 | [`part-02-from-photo-to-glb-simplest-story.md`](part-02-from-photo-to-glb-simplest-story.md) | The app in one paragraph, still without servers or ML jargon. |
| 3 | [`part-03-web-browser-server-http.md`](part-03-web-browser-server-http.md) | Browser, URLs, FastAPI, what “listening on port 8000” means. |
| 4 | [`part-04-jobs-states-queue-mock-vs-real.md`](part-04-jobs-states-queue-mock-vs-real.md) | Why uploads become “jobs”; queued / running / failed; mock icosphere vs real mesh. |
| 5 | [`part-05-ml-concepts-layer-by-layer.md`](part-05-ml-concepts-layer-by-layer.md) | GPU vs CPU; neural nets in one paragraph; diffusion; multi‑view; meshes and GLB — **slowly**. |
| 6 | [`part-06-instantmesh-pipeline-stages.md`](part-06-instantmesh-pipeline-stages.md) | What InstantMesh does step‑by‑step inside the coarse generator. |
| 7 | [`part-07-code-map-which-file-does-what.md`](part-07-code-map-which-file-does-what.md) | Map from disk layout to responsibilities. |
| 8 | [`part-08-chronicle-breakages-fixes-beginner-friendly.md`](part-08-chronicle-breakages-fixes-beginner-friendly.md) | **Everything that broke**, in order; **why**; **how** we fixed it — no shortcuts. |
| 9 | [`part-09-cross-cutting-async-vram-deps-refinement.md`](part-09-cross-cutting-async-vram-deps-refinement.md) | asyncio & GPU locks; VRAM tricks; transformers shims; Trimesh colors; refinement. |
|10 | [`part-10-deployment-colab-vs-huggingface-vs-cloud.md`](part-10-deployment-colab-vs-huggingface-vs-cloud.md) | Colab tunnels, Spaces, paid GPU, ZeroGPU vs T4 — when to use what. |

| **Reference**

- [`GLOSSARY.md`](GLOSSARY.md) — terms from “API” to “vertex color.”
- [`LEARN-MD-MAP.md`](LEARN-MD-MAP.md) — **`learn.md` § ↔ doc part mapping** (read this when searching the long guide).

**Older / narrower logs (still useful)**

- [`../problem_and_sol.md`](../problem_and_sol.md) — original bug‑by‑bug log with commits.
- [`../learn.md`](../learn.md) — transferable patterns for similar projects (companion to the table above).

**Note:** If [`../CONTEXT.md`](../CONTEXT.md) disagrees with code or these docs (e.g. InstantMesh wiring), **trust the repo code and `docs/` first** — `CONTEXT.md` can lag.

---

## Who this is for

- Someone who **can run** the repo but wants to **understand** it.
- Someone who hits an error and needs **mental model**, not only a paste‑this command.

---

## What this folder does *not* replace

- **README.md** at repo root — quick start commands.
- **CONTEXT.md** — high‑level architecture snapshot (may lag; trust `docs/` for nuance).

