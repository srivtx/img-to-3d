# docs2/ — The Ground-Up Learning Path

## What Makes This Different?

**docs/** (written by Claude) = Technical reference for people who already know backend development.  
**docs2/** (this folder) = Educational curriculum for people who don't.

Every part has:
- **Runnable code exercises** you type yourself
- **Why explanations** not just what
- **Multiple angles** on every decision
- **Progressive complexity** — no shortcuts

---

## Learning Path

| Part | Title | What You'll Learn |
|------|-------|-------------------|
| [Part A](part-a-backend-foundations.md) | Backend Foundations | HTTP, sockets, JSON, file uploads, static files — by building them from scratch |
| [Part B](part-b-async-concurrency.md) | Async, Concurrency, Polling | Why blocking is bad, threading vs asyncio, the GIL, event loops, polling vs WebSocket vs SSE |
| [Part C](part-c-ai-inference-deep-dive.md) | AI Inference Line-by-Line | How diffusion creates images, how Zero123 generates views, how LRM reconstructs 3D, how FlexiCubes extracts meshes |
| [Part D](part-d-build-from-scratch.md) | Build-From-Scratch Exercises | 8 hands-on exercises: raw HTTP server, job queue, polling client, Three.js viewer, file uploads, mock AI pipeline, health dashboard, diffusion simulator |
| [Part E](part-e-architecture-debates.md) | Architecture Debates | 10 decision records with pros/cons: mock vs real first, FastAPI vs Flask, polling vs WebSocket, in-memory vs Redis, monolith vs microservices |
| [Part F](part-f-advanced-topics.md) | Advanced Topics | Quantization, distillation, fine-tuning, test-time optimization, NeRF, Gaussian Splatting, ethics, the future |

---

## How to Use These Docs

1. **Read in order.** Each part builds on the previous.
2. **Type the code.** Don't copy-paste. Muscle memory matters.
3. **Break things.** Change values, remove lines, see what happens.
4. **Do the exercises.** They're the actual learning.
5. **Come back later.** These docs work as reference too.

---

## Time Investment

| Part | Reading | Exercises | Total |
|------|---------|-----------|-------|
| A | 1 hour | 2 hours | 3 hours |
| B | 1 hour | 2 hours | 3 hours |
| C | 1.5 hours | 1 hour | 2.5 hours |
| D | 0.5 hours | 4 hours | 4.5 hours |
| E | 1 hour | 0 hours | 1 hour |
| F | 1 hour | 0 hours | 1 hour |
| **Total** | **6 hours** | **9 hours** | **15 hours** |

**15 hours of focused work = solid understanding of the entire stack.**

---

## Prerequisites

- Python installed (3.10+)
- Basic Python knowledge (variables, functions, loops)
- Curiosity and patience

**No 3D experience needed.**  
**No AI experience needed.**  
**No backend experience needed.**

We start from zero.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/srivtx/img-to-3d.git
cd img-to-3d

# Install dependencies
pip install fastapi uvicorn python-multipart

# Read Part A
open docs2/part-a-backend-foundations.md
```

---

## What You'll Be Able to Do After

1. **Explain** how a web server works (not just "it serves pages")
2. **Build** a job queue system with async workers
3. **Implement** polling with exponential backoff
4. **Understand** how diffusion models create images from noise
5. **Describe** the full pipeline from photo to 3D mesh
6. **Compare** 25 different architectures for image-to-3D
7. **Decide** when to use polling vs WebSocket vs SSE
8. **Optimize** models with quantization and distillation
9. **Discuss** the ethical implications of generative AI

---

## The Core Philosophy (Repeated)

> Build one layer at a time. Make it work. Then add the next layer. Complexity is earned, not given.

Our app started as a button that showed a sphere. Then we added uploads. Then progress bars. Then a 3D viewer. Then real AI. Each layer worked before we added the next.

**You can do the same.** Start with Exercise 1 in Part D. Build a raw HTTP server. It takes 20 minutes. Then keep going.

---

## Need Help?

- **Read the code:** `app/main.py`, `app/pipeline/instantmesh.py`, `frontend/app.js`
- **Check the logs:** Server output tells you everything
- **Use the health endpoint:** `GET /health` shows system status
- **Start simple:** If something breaks, test components in isolation

---

## License

These docs are part of the img-to-3d project. See root LICENSE.

---

*Built with care for learners who will become builders.*
