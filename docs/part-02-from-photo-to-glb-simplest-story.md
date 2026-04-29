# Part 02 — From photo to `.glb` (simplest story)

**No servers. No GPUs. Yet.**

## Layer 0 — What problem are we solving?

You have a **2D photograph** — pixels on screen.  
You want a **3D object** you could rotate — something another program (or web viewer) can display.

Rough target: a digital **sculpture** that looks like **the thing in your photo**.

## Layer 1 — What file tells the truth?

Our app emits **GLB**.

- **Plain English:** A `.glb` is a single file packing **geometry** (shape) plus optional **looks** (materials / vertex colors).
- **Why GLB:** Browsers (`Three.js`) can load `.glb` directly to **draw triangles** efficiently.

Think: **triangle soup + paint instructions.**

## Layer 2 — Did the computer invent shape from nowhere?

Roughly yes — partially.

- The neural network pipeline **guess‑fills** unseen sides (you didn’t photograph the backpack’s rear).
- It must **guess** plausible **depth** (“how far bumps stick out”).
- That’s why results can be plausible but **wrong in fine detail**.

We’re not photographing a real object with lasers (that would be a **3D scan** mesh).  
We infer a mesh from statistics learned from datasets.

---

**Next:** how that pipeline talks to you through a browser → Part 03.

