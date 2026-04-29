# Part 03 — Web browser ↔ server (HTTP) without drowning in jargon

**Prerequisites:** [Part 02](part-02-from-photo-to-glb-simplest-story.md).

---

## Layer A — Two programs, one conversation

Think of two people talking:

| Role | In our project | Plain English |
|------|----------------|----------------|
| **Client** | Your **web browser** (Chrome, Firefox, …) | Shows the page, lets you pick a file. |
| **Server** | **FastAPI** app run by **Uvicorn** | Waits for requests, runs Python, sends answers. |

They talk using **HTTP** — a text-based request/response pattern.

---

## Layer B — What is HTTP in one picture?

1. Browser sends: **“POST `/generate-3d`, here is an image file.”**
2. Server sends back: **“200 OK, here is JSON: `{ job_id: … }`.”**

- **POST** ≈ “create something / submit data.”  
- **GET** ≈ “give me something without sending a body” (e.g. `/health`, `/jobs/…`).

You don’t need to memorize headers yet — just: **URL path + method + optional file body.**

---

## Layer C — What is FastAPI?

**FastAPI** is a **Python library** that helps you write the server side:

- You decorate functions with “when someone hits **this path**, run **this code**.”
- It can serve **static files** (HTML/CSS/JS) so the **same process** shows the UI *and* handles API calls.

In this repo, `app/main.py` is that program.

---

## Layer D — What is Uvicorn?

**Uvicorn** is an **ASGI server**: the actual long-running process that:

- **Listens on a port** (e.g. `8000`).
- **Hands requests** to FastAPI.

When docs say *“the server listens on `0.0.0.0:8000`”*:

- **`0.0.0.0`** = accept connections from any network interface (needed in Docker/Colab).
- **`8000`** = port number (like an apartment number for network traffic).

---

## Layer E — Why not “just open an HTML file”?

If you double-click `index.html` from disk, the browser uses the `file://` protocol — **no server**. Many features (uploading to an API, loading models from paths) expect **`http://`**. So we run Uvicorn and visit **`http://localhost:8000`**.

---

## Layer F — Colab + “public URL”

Inside Colab there is a **VM**. Running Uvicorn there gives **`http://localhost:8000` only inside that VM**. A **tunnel** (e.g. Cloudflare) maps a public `https://…` URL to that port so **your laptop’s browser** can reach it.

That’s **not** Hugging Face — it’s “temporary bridge to your temporary machine.”

---

## Layer G — Where this connects to ML

The server doesn’t “magically” know ML. It only:

1. Saves the uploaded image to **disk**.
2. Starts a **job** (see Part 04).
3. Calls Python code that eventually runs **PyTorch on GPU** (much later in the doc series).

**Next:** [Part 04 — Jobs, states, mock vs real](part-04-jobs-states-queue-mock-vs-real.md).
