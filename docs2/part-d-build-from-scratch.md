# Part D: Build-From-Scratch Exercises

## How to Use These Exercises

**Don't read the solutions first.** Try each exercise for at least 30 minutes before looking at hints.

Each exercise builds on the previous ones. Do them in order.

---

## Exercise 1: Build a Raw HTTP Server

**Goal:** Understand what FastAPI hides from you.

### Step 1: Create `raw_server.py`

```python
import socket

# Create a TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Bind to localhost:8000
sock.bind(("localhost", 8000))
sock.listen(5)

print("Server listening on http://localhost:8000")

while True:
    # Accept a connection
    conn, addr = sock.accept()
    print(f"Connection from {addr}")
    
    # Read the request (up to 4096 bytes)
    request = conn.recv(4096).decode("utf-8")
    print("Request received:")
    print(request[:500])
    
    # Parse the request line
    lines = request.split("\r\n")
    if lines:
        method, path, version = lines[0].split(" ")
        print(f"Method: {method}, Path: {path}")
    
    # Build a response
    body = f"<h1>You requested: {path}</h1>"
    response = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
        f"{body}"
    )
    
    # Send response
    conn.send(response.encode("utf-8"))
    conn.close()
```

**Run it:** `python raw_server.py`  
**Test it:** Open `http://localhost:8000/hello` in your browser.

### Step 2: Add Routing

Modify the server to handle different paths:
- `/` → "Welcome"
- `/time` → Current time
- `/json` → `{"status": "ok"}`
- Anything else → 404

### Step 3: Handle POST Requests

Add a `/echo` endpoint that:
1. Reads the POST body
2. Returns it back to the client

**Hint:** POST requests have a body after the headers. Look for `Content-Length` to know how much to read.

---

## Exercise 2: Build a Job Queue System

**Goal:** Understand how our async queue works.

### Step 1: In-Memory Queue

```python
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
import uuid
import time

class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    id: str
    status: Status
    created_at: float
    result: Optional[str] = None

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
    
    def create_job(self) -> Job:
        job = Job(
            id=str(uuid.uuid4())[:8],
            status=Status.PENDING,
            created_at=time.time()
        )
        self.jobs[job.id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

# Test
queue = JobQueue()
job = queue.create_job()
print(f"Created job: {job.id}")
print(f"Status: {queue.get_job(job.id).status.value}")
```

### Step 2: Make It Async and Thread-Safe

Add `asyncio.Lock()` so multiple coroutines can safely use the queue simultaneously.

### Step 3: Add Background Workers

Write a worker that:
1. Takes jobs from the queue
2. Processes them (simulate with `asyncio.sleep`)
3. Updates status
4. Limits concurrent workers (use `asyncio.Semaphore`)

### Step 4: Build an API Around It

Create a FastAPI app with:
- `POST /jobs` → Create job
- `GET /jobs/{id}` → Get status
- Background worker processing jobs

---

## Exercise 3: Build a Polling Client

**Goal:** Understand the frontend's perspective.

### Step 1: Simple Polling

```python
import requests
import time

class PollingClient:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def submit_job(self, data):
        """Submit a job and return job ID."""
        response = requests.post(f"{self.base_url}/jobs", json=data)
        return response.json()["job_id"]
    
    def poll_until_done(self, job_id, interval=1.0):
        """Poll until job completes."""
        while True:
            response = requests.get(f"{self.base_url}/jobs/{job_id}")
            status = response.json()
            
            print(f"Status: {status['status']}, Progress: {status.get('progress', 0)}%")
            
            if status["status"] in ("completed", "failed"):
                return status
            
            time.sleep(interval)

# Use it
client = PollingClient("http://localhost:8000")
job_id = client.submit_job({"task": "process_image"})
result = client.poll_until_done(job_id)
print(f"Final result: {result}")
```

### Step 2: Add Exponential Backoff

Modify `poll_until_done` to:
- Start with 0.5s interval
- Increase by 1.5x each poll
- Cap at 10 seconds

### Step 3: Add Progress Callback

```python
def poll_with_callback(self, job_id, on_progress):
    """Call on_progress(percent, message) each poll."""
    ...

# Usage
def show_progress(percent, message):
    bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
    print(f"\r[{bar}] {percent}% {message}", end="")

client.poll_with_callback(job_id, show_progress)
```

---

## Exercise 4: Build a Minimal Mesh Viewer

**Goal:** Understand Three.js basics.

### Step 1: HTML Canvas

```html
<!DOCTYPE html>
<html>
<head>
    <title>Mesh Viewer</title>
    <style>
        body { margin: 0; overflow: hidden; }
        canvas { display: block; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <script type="module">
        // Your code here
    </script>
</body>
</html>
```

### Step 2: Draw a Triangle

```javascript
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl');

// Vertex shader
const vsSource = `
    attribute vec4 position;
    void main() {
        gl_Position = position;
    }
`;

// Fragment shader
const fsSource = `
    void main() {
        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red
    }
`;

// Compile shaders, link program, create buffer, draw
// (This is verbose — that's why we use Three.js)
```

### Step 3: Use Three.js Instead

```javascript
import * as THREE from 'three';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Create a cube
const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
const cube = new THREE.Mesh(geometry, material);
scene.add(cube);

camera.position.z = 5;

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;
    renderer.render(scene, camera);
}
animate();
```

### Step 4: Load a GLB File

```javascript
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
loader.load('model.glb', (gltf) => {
    scene.add(gltf.scene);
});
```

### Step 5: Add Orbit Controls

```javascript
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
```

---

## Exercise 5: Build a File Upload System

**Goal:** Understand multipart form data.

### Step 1: HTML Form

```html
<form id="uploadForm">
    <input type="file" id="fileInput" accept="image/*">
    <button type="submit">Upload</button>
</form>
<div id="status"></div>

<script>
document.getElementById('uploadForm').onsubmit = async (e) => {
    e.preventDefault();
    
    const file = document.getElementById('fileInput').files[0];
    const formData = new FormData();
    formData.append('image', file);
    
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    document.getElementById('status').textContent = result.message;
};
</script>
```

### Step 2: FastAPI Handler

```python
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/upload")
async def upload(image: UploadFile = File(...)):
    contents = await image.read()
    
    # Save to disk
    with open(f"uploads/{image.filename}", "wb") as f:
        f.write(contents)
    
    return {
        "filename": image.filename,
        "size": len(contents),
        "message": f"Saved {len(contents)} bytes"
    }
```

### Step 3: Show Upload Progress

Add a progress bar using `XMLHttpRequest` instead of `fetch`:

```javascript
const xhr = new XMLHttpRequest();

xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
        const percent = (e.loaded / e.total) * 100;
        console.log(`Upload: ${percent}%`);
    }
};

xhr.open('POST', '/upload');
xhr.send(formData);
```

---

## Exercise 6: Build a Mock AI Pipeline

**Goal:** Understand our architecture without real AI.

### Requirements

Build a system where:
1. User uploads an image
2. Server "processes" it (just sleeps for 5 seconds)
3. Server returns a placeholder sphere
4. User sees the sphere in a 3D viewer

### Components

**Backend (FastAPI):**
- `POST /generate` → Accept image, create job, return job_id
- `GET /jobs/{id}` → Return status
- Background task that "processes" the image

**Frontend:**
- Upload form
- Status display (polling)
- Three.js viewer (show sphere)

**The "sphere":**
```python
import trimesh
sphere = trimesh.creation.icosphere(subdivisions=3)
sphere.export("output.glb")
```

### Challenge: Add "Progress"

During the 5-second sleep, update progress from 0% to 100% in 10 steps.

---

## Exercise 7: Build a Health Check Dashboard

**Goal:** Understand monitoring.

Build a simple dashboard that shows:
- Server uptime
- Number of jobs queued/running/completed
- Device (CPU/GPU)
- Memory usage

**Backend endpoint:**
```python
import psutil

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "uptime": time.time() - start_time,
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "jobs": {
            "queued": len([j for j in jobs.values() if j.status == "pending"]),
            "running": len([j for j in jobs.values() if j.status == "running"]),
            "completed": len([j for j in jobs.values() if j.status == "completed"]),
        }
    }
```

**Frontend:**
Simple HTML page that fetches `/health` every 5 seconds and displays the data.

---

## Exercise 8: Build a Simple Diffusion Simulator

**Goal:** Intuitively understand how diffusion works.

### The Exercise

Write a Python script that:

1. Loads an image
2. Adds random noise to it (forward diffusion)
3. Saves 10 versions with increasing noise
4. Then "denoises" by averaging with the original (fake reverse diffusion)

```python
import numpy as np
from PIL import Image

def add_noise(image, amount):
    """Add random noise to image."""
    noise = np.random.normal(0, amount, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Load image
img = np.array(Image.open("photo.jpg"))

# Forward diffusion: add noise progressively
for i in range(10):
    amount = i * 25  # More noise each step
    noisy = add_noise(img, amount)
    Image.fromarray(noisy).save(f"noise_{i}.png")

# Reverse diffusion: fake denoising by blending with original
noisy = add_noise(img, 250)
for i in range(10):
    alpha = (i + 1) / 10  # Blend factor
    denoised = noisy * (1 - alpha) + img * alpha
    Image.fromarray(denoised.astype(np.uint8)).save(f"denoise_{i}.png")
```

**Observe:**
- At high noise, the image is unrecognizable
- "Denoising" by blending with original is cheating (we know the answer)
- Real diffusion models learn to denoise without seeing the original

---

## Solutions and Hints

### Exercise 1 Hints
- HTTP requests end with `\r\n\r\n`
- POST body comes after the headers
- Use `Content-Length` header to know how many bytes to read

### Exercise 2 Hints
- `asyncio.Lock()` ensures only one coroutine modifies the dictionary at a time
- `asyncio.Semaphore(n)` allows at most `n` concurrent workers
- Use `asyncio.create_task()` to fire-and-forget background work

### Exercise 3 Hints
- Exponential backoff: `interval = min(interval * 1.5, max_interval)`
- For progress callback, pass a function that gets called each iteration
- Consider adding a timeout (max total polling time)

### Exercise 4 Hints
- Three.js uses a right-handed coordinate system (Y is up)
- `requestAnimationFrame` schedules the next frame
- OrbitControls needs to be updated each frame (`controls.update()`)

### Exercise 5 Hints
- `FormData` automatically sets `Content-Type: multipart/form-data` with boundary
- FastAPI handles multipart parsing automatically
- For progress, XMLHttpRequest is more flexible than fetch()

### Exercise 6 Hints
- Use `BackgroundTasks` in FastAPI
- Update job status from the background task
- The sphere can be generated once and cached

### Exercise 7 Hints
- `psutil` gives system info
- Store `start_time = time.time()` when server starts
- Use JavaScript `setInterval` to refresh the dashboard

### Exercise 8 Hints
- `np.random.normal(0, std, shape)` generates Gaussian noise
- `np.clip` ensures pixel values stay in [0, 255]
- Real diffusion uses neural networks, not simple blending

---

**Next:** Part E — Architecture Debates and Decision Records
