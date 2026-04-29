# Part 3: How the Web Works — Browsers, Servers, and APIs

## The Restaurant Analogy

Understanding web architecture is easier with an analogy.

### The Restaurant

- **You (Customer)** = The web browser
- **Waiter** = HTTP (the protocol that carries requests)
- **Kitchen** = The server (FastAPI in our case)
- **Menu** = The API (list of what you can order)
- **Chef** = The AI model (does the hard work)
- **Food** = The 3D model file

**You don't talk directly to the chef.** You tell the waiter what you want. The waiter brings it to the kitchen. The kitchen does the work. The food comes back.

---

## What Happens When You Open Our App

### Step 1: You Type a URL

```
https://your-app.com/
```

**What happens behind the scenes:**
1. Browser asks DNS: "What's the IP address for your-app.com?"
2. DNS responds: "It's 203.0.113.42"
3. Browser opens a TCP connection to that IP
4. Browser sends an HTTP GET request

### Step 2: The Server Responds

Our FastAPI server receives:
```http
GET / HTTP/1.1
Host: your-app.com
```

FastAPI looks at the `/` route and returns `index.html`.

The response is:
```http
HTTP/1.1 200 OK
Content-Type: text/html

<!DOCTYPE html>
<html>...</html>
```

### Step 3: Browser Builds the Page

The browser:
1. Parses HTML
2. Sees `<link rel="stylesheet" href="/static/style.css">`
3. Makes ANOTHER request: `GET /static/style.css`
4. Sees `<script type="module" src="/static/app.js">`
5. Makes ANOTHER request: `GET /static/app.js`
6. Sees Three.js importmap
7. Fetches Three.js from unpkg.com

**Key point:** A single page load = many requests.

### Step 4: You Upload a Photo

You drag a photo into the browser.

The JavaScript:
1. Reads the file
2. Creates a `FormData` object
3. Sends: `POST /generate-3d` with the image

```javascript
fetch('/generate-3d', {
    method: 'POST',
    body: formData
})
```

### Step 5: The Server Processes

FastAPI receives:
```http
POST /generate-3d HTTP/1.1
Content-Type: multipart/form-data

[binary image data]
```

The server:
1. Saves the image to disk
2. Creates a job ID
3. Starts a background task
4. Immediately returns: `{ "job_id": "abc123", "status": "pending" }`

### Step 6: The Browser Polls

The JavaScript doesn't wait. It starts polling:
```javascript
setInterval(() => {
    fetch('/jobs/abc123')
        .then(r => r.json())
        .then(status => updateUI(status))
}, 1000)
```

Every second, it asks: "Is it done yet?"

### Step 7: The Server Finishes

When the mesh is ready, the next poll returns:
```json
{
    "job_id": "abc123",
    "status": "completed",
    "preview_url": "/outputs/abc123/preview.glb",
    "final_url": "/outputs/abc123/final.glb",
    "progress_percent": 100
}
```

### Step 8: The Browser Loads the 3D Model

Three.js loads the GLB:
```javascript
loader.load('/outputs/abc123/preview.glb', (gltf) => {
    scene.add(gltf.scene)
})
```

---

## Key Concepts Explained

### HTTP Methods

| Method | Meaning | Example |
|--------|---------|---------|
| **GET** | Read something | Load a page, check status |
| **POST** | Create something | Upload an image |
| **PUT** | Update something | Replace a file |
| **DELETE** | Remove something | Delete a job |

### Status Codes

| Code | Meaning | When You See It |
|------|---------|----------------|
| **200** | OK | Everything worked |
| **201** | Created | Resource was created |
| **400** | Bad Request | You sent bad data |
| **404** | Not Found | URL doesn't exist |
| **500** | Server Error | The server crashed |

### JSON

**JSON** (JavaScript Object Notation) is the universal language of APIs.

```json
{
    "job_id": "abc123",
    "status": "completed",
    "progress": 100
}
```

**Why JSON?**
- Human-readable
- Every programming language can parse it
- Simple structure: objects `{}` and arrays `[]`

### REST API

**REST** = Representational State Transfer. It's a style of designing APIs.

Our API is RESTful:
- `POST /generate-3d` → Create a job
- `GET /jobs/abc123` → Read job status
- `GET /health` → Read system health

### CORS (Cross-Origin Resource Sharing)

Browsers have a security rule: **a page from one domain can't make requests to another domain unless allowed.**

Example:
- Your frontend is at `https://my-app.com`
- Your API is at `https://api.my-app.com`
- The browser blocks the request unless the API says "it's okay"

**Our fix:** We add a CORS middleware:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow everyone (for development)
)
```

### Static Files

Static files = files that don't change (CSS, JS, images, 3D models).

FastAPI serves them directly:
```python
app.mount("/static", StaticFiles(directory="frontend"))
app.mount("/outputs", StaticFiles(directory="outputs"))
```

When the browser requests `/static/style.css`, FastAPI reads `frontend/style.css` and returns it.

### Ports

A **port** is like a door number on a computer.

```
localhost:8000  # Door 8000
localhost:7860  # Door 7860 (Hugging Face)
```

**Why ports?** One computer can run many services. Each service listens on a different port.

Our server listens on port 8000 by default.

### Localhost

`localhost` = "this computer."

```
http://localhost:8000/  # Talk to the server on THIS machine
```

**Why localhost matters:** During development, everything runs on your own computer. You don't need the internet.

---

## Async vs Sync

### Synchronous (Blocking)
```python
def process_image(image):
    result = run_ai_model(image)  # Wait here until done (5 minutes)
    return result
```

**Problem:** While waiting, the server can't handle other requests.

### Asynchronous (Non-Blocking)
```python
async def process_image(image):
    result = await run_ai_model(image)  # Let other requests run while waiting
    return result
```

**Solution:** The server can handle thousands of requests simultaneously.

### Our Approach

We use a hybrid:
1. **API layer:** Async (FastAPI handles many requests)
2. **AI inference:** Sync but in a background thread (doesn't block the API)

```python
@app.post("/generate-3d")
async def upload(image):
    job = create_job()
    background_tasks.add_task(process_job, job.id)  # Runs in background
    return {"job_id": job.id}  # Returns immediately
```

---

## The Full Request Lifecycle

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Browser   │  HTTP   │   FastAPI    │ Thread  │   AI Model  │
│  (Frontend) │  ←──→   │  (Backend)   │  ───→   │  (GPU)      │
└─────────────┘         └──────────────┘         └─────────────┘
      │                        │                        │
      │  POST /generate-3d     │                        │
      │ ─────────────────────> │                        │
      │                        │  Create job            │
      │                        │  Start background task │
      │  {job_id: "abc123"}    │                        │
      │ <───────────────────── │                        │
      │                        │                        │
      │  GET /jobs/abc123      │                        │
      │ ─────────────────────> │                        │
      │  {status: "pending"}   │                        │
      │ <───────────────────── │                        │
      │      (repeat)          │                        │
      │                        │                        │
      │                        │        Process image   │
      │                        │ ─────────────────────> │
      │                        │                        │
      │  GET /jobs/abc123      │                        │
      │ ─────────────────────> │                        │
      │  {status: "completed", │                        │
      │   preview_url: "..."}  │                        │
      │ <───────────────────── │                        │
      │                        │                        │
      │  GET /outputs/abc123/  │                        │
      │       preview.glb      │                        │
      │ ─────────────────────> │                        │
      │  [binary GLB data]     │                        │
      │ <───────────────────── │                        │
      │                        │                        │
      │  [Three.js renders]    │                        │
```

---

## Summary

| Concept | Simple Definition |
|---------|-------------------|
| **HTTP** | The language browsers and servers use to talk |
| **GET/POST** | Read data / Send data |
| **Status Code** | Did it work? (200 = yes, 404 = no, 500 = oops) |
| **JSON** | Text format for sending structured data |
| **API** | List of operations a server offers |
| **CORS** | Security rule about cross-domain requests |
| **Static File** | A file that doesn't change (CSS, JS, images) |
| **Port** | Door number for a network service |
| **Async** | Don't wait, handle other things while processing |
| **Background Task** | Work that happens after the immediate response |

---

**Next:** [Part 4: Our Architecture](part-04-our-architecture.md) — We explain exactly what we built, piece by piece.
