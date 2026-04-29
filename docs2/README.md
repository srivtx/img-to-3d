# docs2/ — The Complete Beginner's Guide to Image-to-3D

## What is this folder?

This is the **deep, slow, no-shortcuts** explanation of everything we built, everything that broke, and everything we learned.

**docs/** (the sibling folder) was written by Claude. It's excellent and technical.  
**docs2/** is written by us — it's the version we wish we had on day zero, when we didn't know what a GPU was.

---

## How to read this

**Don't skip ahead.** Each part builds on the previous one.

| Part | Title | What you'll understand |
|------|-------|----------------------|
| [Part 0](part-00-philosophy.md) | Philosophy: Why We Go Slow | Why complexity must be earned, not given |
| [Part 1](part-01-what-is-3d.md) | What is 3D? | Vertices, polygons, meshes, files — from zero |
| [Part 2](part-02-what-is-ai.md) | What is AI/ML? | Neural networks, diffusion, feed-forward — explained simply |
| [Part 3](part-03-the-web.md) | How the Web Works | Browsers, servers, APIs, HTTP — the plumbing |
| [Part 4](part-04-our-architecture.md) | What We Built | The full architecture, explained piece by piece |
| [Part 5](part-05-everything-that-broke.md) | Everything That Broke | Every error, why it happened, how we fixed it |
| [Part 6](part-06-alternative-architectures.md) | 20+ Other Ways to Build This | Architecture alternatives from beginner to advanced |
| [Part 7](part-07-glossary.md) | Glossary | Every term defined simply |

---

## Who this is for

- **Absolute beginners** who've never touched 3D or AI
- **Developers** who know Python but not ML pipelines
- **Your future self** who will forget why you made every decision
- **Anyone who inherits this project** and needs to understand it deeply

---

## The core idea of this project

**Traditional 3D creation:** Hours in Blender, sculpting vertex by vertex.  
**Our approach:** Upload one photo. Get a 3D model in 10 seconds.

We don't generate perfect 3D in one step. Instead:
1. **Coarse model** — fast, approximate, ugly but recognizable (~2 seconds)
2. **Refinement** — improve shape and smoothness in background (~10 seconds)
3. **Progressive delivery** — user sees something immediately, quality improves over time

This is called a **coarse-to-fine pipeline**.

---

## The hard truth we learned

We spent days trying to make this work. We learned that:

1. **AI models are easy to find, hard to wire** — InstantMesh exists, but connecting it to a web server is 90% of the work
2. **The error is never where you think** — We thought the model was broken. Actually, our import path was wrong.
3. **Subprocess is often better than import** — Running a Python script as a command is more reliable than trying to import its modules
4. **GPU memory is precious** — 16GB sounds like a lot until you load a 4GB model, 2GB of dependencies, and 8GB of intermediate tensors
5. **Web + ML = two different worlds** — The browser speaks JavaScript. The GPU speaks CUDA. Making them talk is the whole job.

---

Let's begin.
