# Part 04 — Jobs, states, and why “mock” exists

**Prerequisites:** [Part 03](part-03-web-browser-server-http.md).

---

## Layer A — Why not block the browser for 90 seconds?

Turning a photo into a **real** 3D mesh can take **~60–90 seconds** on a good GPU.

If the server made you **wait** with a frozen tab until done, the browser might **time out** and you’d have no progress UI.

So the app uses a **job pattern**:

1. **POST `/generate-3d`** → server answers **immediately** with a **`job_id`** (small JSON).
2. Browser **polls** **GET `/jobs/{job_id}`** every second or so until status is **done** or **failed**.

You see a **progress bar** in the UI while polling — that percentage is wired to coarse vs refine stages in code, but the important idea is: **HTTP stays responsive.**

---

## Layer B — States (mental model)

In **this codebase** the API uses explicit enum values (see `app/core/models.py`):

```
pending → processing_coarse → coarse_ready → refining → completed
                                              ↘ failed
```

(In plain English you can still think “queued → running → done,” but the JSON will say **`processing_coarse`**, **`refining`**, etc.)

- **`pending`:** upload saved; background task not started or job just created.
- **`coarse_ready`:** **`preview.glb`** path known; refinement may start next.
- **`refining`:** smoothing / simplification for **`final.glb`**.
- **`completed` / `failed`:** terminal — polling can stop (**`completed`** means both coarse and refinement finished successfully).


## Layer C — What is “coarse” vs “refine”?

| Stage | Output | Purpose |
|-------|--------|---------|
| **Coarse** | `preview.glb` | Primary 3D from InstantMesh (or mock). Fast enough to preview. |
| **Refinement** | `final.glb` | Optional mesh cleanup (smooth, optionally simplify). |

So one upload can produce **two files** — both are real GLB files; “coarse” does **not** mean “fake.”

---

## Layer D — What is “mock” and why does it exist?

**Mock mode** means: *we intentionally skip the heavy neural network* and **generate a simple placeholder shape** (in our case a sphere made of triangles — an **icosphere**) so **the rest of the app still works** (queue, GLB download, Three.js viewer).

**Why have it at all?**

- Developers on **laptops without CUDA** shouldn’t require **four gigabytes** of weights to **boot the server**.
- If something breaks at install time, **the API still responds** instead of refusing to start.

**The danger:** mock looks “fine” until you inspect the mesh (round ball vs real backpack). The **`CoarseGenerator`** singleton sets **`_using_mock`** when load or inference degrades — watch **`[InstantMesh]`** stderr lines and **`/health`** (`models_loaded` only confirms load finished, not that you’re non-mock unless you inspect generator state in logs).

**How it’s triggered (simplified):**

- Environment variable **`USE_INSTANTMESH=true`** tries to load real InstantMesh inside `CoarseGenerator.load()`.
- If **any exception** is raised during that load → code sets **“use mock.”**
- Similarly, **per-job** inference can exception → fall back to mock **for that job only.**

---

## Layer E — In-memory queue

This project uses an **in-memory asyncio queue**: jobs disappear if the server **process exits**.

That’s acceptable for demos; **production** often uses Redis / a database — out of scope here, but know the limitation.

---

## Layer F — Design choice: degrade silently vs crash

Catching exceptions and falling back **keeps demos running** but can **hide bugs** behind a sphere shape. Mitigations we adopted:

- Smoke tests **before** server start (`import_module`).
- Loud logs **`[InstantMesh]`** with **`flush`** where possible.
- Understanding **deepest-visible-failure** (see Part 08).

---

**Next:** ML vocabulary built slowly → [Part 05](part-05-ml-concepts-layer-by-layer.md).
