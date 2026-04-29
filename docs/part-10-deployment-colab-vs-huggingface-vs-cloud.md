# Part 10 — Deployment: Colab, Hugging Face Spaces, and beyond

**Prerequisites:** [Part 03](part-03-web-browser-server-http.md) (what a **port** and **tunnel** mean).

---

## Layer A — Three different “we run a server” stories

| Environment | Who owns the machine? | GPU? | URL | Best for |
|---------------|-------------------------|------|-----|----------|
| **Your laptop / Docker locally** | You | Sometimes (Apple / NVIDIA) | `http://localhost:8000` | Dev iteration, CPU mock |
| **Google Colab** | Google (temporary VM) | Often **T4** free tier | **`localhost` inside VM** — needs **tunnel** for phone/laptop browser | Demos, teaching, cheap GPU bursts |
| **Hugging Face Spaces** | HF | CPU free; **paid GPU tiers** | **`https://*.hf.space`** public | Sharing, persistence options, minimal ops |

None of these **replace** each other — they differ in **billing**, **cold start**, **disk persistence**, and **VRAM class**.

---

## Layer B — Colab path (this repo)

**`generate_colab.py`** emits **`colab/Image_to_3D_Generator.ipynb`**.

Typical flow:

1. Clone repo, **pip install** pins (torch, diffusers, **`onnxruntime`**, …).
2. Download **multi-gigabyte checkpoints** once (slow but cached for the VM session).
3. Run **pre-flight imports** (**`import_module`**, not **`find_spec`**).
4. Start **Uvicorn** (often with **`PYTHONUNBUFFERED`**).
5. Start **Cloudflare tunnel** (`cloudflared`) so **`https://trycloudflare.com/...`** maps to **`localhost:8000`**.

**Why tunnel?** Your browser is **not** on Colab’s VM — `localhost` on your laptop **is not** Colab’s `localhost`.

**Caveat:** VM **dies** → **weights gone** unless you use persistent drive / re-download.

---

## Layer C — Hugging Face Spaces path

See root **`HUGGINGFACE_DEPLOY.md`**.

- **Docker** SDK Space runs your **`Dockerfile`**.
- **CPU tier** can **boot the app** (mock / light paths) for **UI testing**.
- **Real InstantMesh** usually needs **GPU tier** + **storage** for weights (multi-GB).

**Trade-off:** less tunnel fiddling than Colab; **cost** and **queueing** depend on HF’s scheduling.

---

## Layer D — “Cloud generic” (AWS / GCP / RunPod / …)

Pattern stays the same:

1. **GPU instance** with drivers + CUDA-matched PyTorch wheel.
2. **Open port** (or reverse proxy) for HTTP.
3. **Persistent volume** for model weights (don’t re-download every boot).
4. **Horizontal scaling** is **hard** for big GPU models — often **one job per GPU** queues.

---

## Layer E — Choosing default model config

- **`instant-mesh-base`** — default for **~16 GB** class GPUs.
- **`instant-mesh-large`** — when you have **headroom** (A100, large consumer card with 24 GB+).

Document your choice in deployment notes so operators don’t assume “large == better” on every card.

---

## Layer F — Where Part 08’s fixes apply

| Fix area | Colab | HF Space | Cloud VM |
|----------|-------|----------|------------|
| **`onnxruntime` in requirements** | ✓ | ✓ | ✓ |
| **Vendored `zero123plus` path** | ✓ | ✓ | ✓ |
| **Transformers shims** | ✓ | ✓ | ✓ |
| **VRAM / base config / offload** | ✓ (T4) | ✓ (depends on tier) | ✓ |

---

**You’ve reached the end of the ladder.** Return to [README](README.md) or [`GLOSSARY.md`](GLOSSARY.md) for quick definitions.
