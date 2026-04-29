# Part 01 — Reading order & philosophy

## Why “step by step” matters here

Machine learning web apps pile **five different specialties** onto one codebase:

1. **Web:** browser, HTTP, files.
2. **Backend:** web server, background work, concurrency.
3. **Systems:** OS processes, GPUs, RAM vs VRAM.
4. **Software packaging:** pip, optional dependencies, broken imports.
5. **ML:** models whose training stack is older than your Python packages.

Each layer introduces **failure modes** that surface as “it doesn’t work” in the UI. If we jump straight to “instantmesh failed at line 457,” you skip the scaffolding that explains **why line 457 even runs**.

These docs deliberately:

- Start **without jargon** (“a program turns your photo into a file”).
- Add **only one new idea per part** (“now we add HTTP”).
- Separate **mental models** (“what diffusion is”) from **bug archaeology** (“rembg needed onnxruntime”).
- Treat **fixes as cumulative** — each fix unblocked the **next deeper** error (the “deepest visible failure” pattern).

---

## Complexity ramp (mental map)

```
Part 02  →  Goal in plain English (output file exists)
Part 03  →  Two programs talk (browser + server)
Part 04  →  Long work is split into jobs + states (+ mock fallback)
Part 05  →  Enough ML vocabulary to read logs
Part 06  →  Map vocabulary to InstantMesh’s actual stages
Part 07  →  Repo layout ↔ responsibility
Part 08  →  Timeline of failures (the “lesson” narrative)
Part 09  →  Engineering patterns reused across fixes
Part 10  →  Where compute can live + tradeoffs
```

You may skip ahead if you already know web basics — but Parts **05–06** and **08** are where misunderstood terms usually hide.

---

## A note on honesty

Production ML tutorials often show:

> “Five lines import model; congratulations.”

Reality involves **hidden dependencies**, **version skew**, **GPU memory cliffs**, **buffered subprocess logs**. Part 08 describes **real** failures so you normalize them — they signal **progress**, not personal failure.

