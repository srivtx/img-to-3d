# How `learn.md` maps onto this `docs/` series

The root file **`learn.md`** (long, many sections — written as transferable patterns) is our **dense reference.** The **`docs/part-*.md`** files are the **guided path** — same ideas, slower ramp.

Use this map when **Ctrl+F** hunting in **`learn.md`**.

---

| `learn.md` section | Teach here first |
|--------------------|------------------|
| **§1 Debugging — deepest visible failure** | [Part 08](part-08-chronicle-breakages-fixes-beginner-friendly.md) Steps 1–3, 11 |
| **§1 Buffered stdout** | Part 08 Step 2; [Part 09](part-09-cross-cutting-async-vram-deps-refinement.md) §6 |
| **§1 Broad except** | Part 08 Step 1 |
| **§1 Pre-flight checks** | [Part 04](part-04-jobs-states-queue-mock-vs-real.md) Layer F; Part 09 §6 |
| **§2 Imports — find_spec vs import_module** | Part 08 Step 3 |
| **§2 sys.path for vendored code** | [Part 07](part-07-code-map-which-file-does-what.md) + `instantmesh.py` mentally |
| **§2 Monkey-patch class propagation** | Part 08 Step 5 |
| **§3 Library drift — pinning vs shimming** | Part 08 Steps 5–6; Part 09 §4 |
| **§3 Version-pinned mirror trap** | Part 08 Step 4 |
| **§3 Optional extras (onnxruntime)** | Part 08 Step 3 |
| **§4 GPU VRAM math** | Part 08 Step 6; Part 09 §3 |
| **§4 expandable_segments** | Part 08 Step 6; Part 09 §3b |
| **§4 Offload pipeline** | Part 08 Step 7; Part 09 §3c |
| **§4 fp16 + CPU warnings** | Part 09 §3d |
| **§5 Trimesh visuals / destructive UV** | Part 09 §5; Part 08 Step 8 |
| **§6 FastAPI async / asyncio.to_thread** | Part 09 §1–2 |
| **§6 GPU threading lock pattern** | Part 09 §1 (mention) |
| **§7 Colab / notebook regeneration** | [Part 10](part-10-deployment-colab-vs-huggingface-vs-cloud.md) §B |
| **§8 InstantMesh four-stage arc** | [Part 06](part-06-instantmesh-pipeline-stages.md); learn §8 for extra depth |
| **§9 using_mock observability suggestion** | Part 08 Step 11; Part 04 (optional health extension) |

---

## Cross-reference: chronological bug log vs patterns

| Source | Angle |
|--------|-------|
| **`problem_and_sol.md`** | **Exact** six failures & commits — engineer-facing log |
| **`docs/part-08-...`** | Same story — **tutorial voice**, domino intuition |
| **`learn.md`** | **Portable patterns** for *future* repos (beyond this bug list) |

---

## When to read which

1. **First day:** Parts **01→05** (`docs/README`).
2. **Debugging “icosphere hell”:** Part **08** + **`problem_and_sol.md`** appendix numbers.
3. **Polishing infra:** **`learn.md` §7 + §9** + Part **10**.

---

*If `CONTEXT.md` still says InstantMesh is “not wired,” treat that file as stale — follow **code + `docs/part-*.md`**.*
