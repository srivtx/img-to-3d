# Part A: Python Backend Foundations

## How to Use This Document

**Don't just read it. Type the code.**

Every section has exercises. Do them. Make mistakes. Fix them. That's how learning works.

---

## Chapter 1: What is a Server? (The Absolute Basics)

### The Analogy: A Restaurant

Imagine a restaurant:
- **Customer** = Your web browser
- **Waiter** = The network connection
- **Kitchen** = The server (our Python program)
- **Menu** = The API (what operations are available)
- **Food** = The response data

The customer doesn't walk into the kitchen. They talk to the waiter. The waiter talks to the kitchen. The kitchen does the work. The food comes back.

**A server is just a program that waits for requests and sends responses.**

### Exercise 1.1: Your First Server (Type This)

Create a file called `server1.py`:

```python
import socket

# Create a socket (the "door" that listens for connections)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind to port 8000 on localhost
server_socket.bind(('localhost', 8000))

# Listen for incoming connections (max 5 queued)
server_socket.listen(5)

print("Server listening on http://localhost:8000")
print("Open your browser and visit that URL")
print("Press Ctrl+C to stop")

while True:
    # Wait for someone to connect
    client_socket, address = server_socket.accept()
    print(f"Someone connected from {address}")
    
    # Read what they sent (up to 1024 bytes)
    request = client_socket.recv(1024).decode('utf-8')
    print(f"They said:\n{request[:200]}...")
    
    # Send a response (HTTP format)
    response = """HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<h1>Hello! You reached my server.</h1>"""
    client_socket.send(response.encode('utf-8'))
    
    # Close the connection
    client_socket.close()
```

Run it:
```bash
python server1.py
```

Open your browser to `http://localhost:8000`

**What you should see:** A page that says "Hello! You reached my server."

**What just happened:**
1. Python created a network socket
2. It bound to port 8000 (claimed that "door number")
3. It listened for connections
4. Your browser connected and sent an HTTP request
5. Python sent back an HTTP response
6. The browser rendered the HTML

### Exercise 1.2: Understanding the Request

Modify `server1.py` to print the FULL request (remove `[:200]`):

```python
print(f"They said:\n{request}")
```

Restart the server. Visit it again. Look at what your browser sent:

```
GET / HTTP/1.1
Host: localhost:8000
User-Agent: Mozilla/5.0...
Accept: text/html,application/xhtml+xml...
Accept-Language: en-US,en;q=0.9
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
```

**This is HTTP.** It's just text. The browser sends text. The server sends text back.

**Questions to answer:**
1. What does `GET` mean?
2. What does `/` mean?
3. What is `User-Agent`?
4. What is `Connection: keep-alive`?

**Answers:**
1. `GET` = "I want to read something"
2. `/` = "the root path" (like the homepage)
3. `User-Agent` = "I'm Chrome version X on MacOS"
4. `keep-alive` = "don't close the connection yet, I might send more requests"

### Exercise 1.3: Different Paths

Modify the server to handle different URLs:

```python
while True:
    client_socket, address = server_socket.accept()
    request = client_socket.recv(1024).decode('utf-8')
    
    # Parse the request line (first line)
    request_line = request.split('\n')[0]
    method, path, version = request_line.split(' ')
    print(f"Method: {method}, Path: {path}")
    
    # Route to different content based on path
    if path == '/':
        body = "<h1>Home</h1>"
    elif path == '/about':
        body = "<h1>About</h1><p>This is my server.</p>"
    elif path == '/json':
        body = '{"status": "ok", "message": "Hello from JSON"}'
    else:
        body = "<h1>404 Not Found</h1>"
    
    response = f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n{body}"
    client_socket.send(response.encode('utf-8'))
    client_socket.close()
```

**Test these URLs:**
- `http://localhost:8000/` → Home
- `http://localhost:8000/about` → About
- `http://localhost:8000/json` → JSON response
- `http://localhost:8000/anything-else` → 404

**This is routing.** Different URLs → different code runs.

---

## Chapter 2: Why We Use FastAPI (Instead of Raw Sockets)

Raw sockets work, but they're tedious. For every endpoint, you write:
- Parse the request
- Extract the path
- Check the method
- Parse query parameters
- Handle errors
- Format the response
- Set content-type headers

FastAPI does all of this for us.

### Exercise 2.1: The Same Server in FastAPI

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from FastAPI"}

@app.get("/about")
def about():
    return {"message": "This is my server"}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id, "name": f"User {user_id}"}
```

Save as `server_fastapi.py` and run:
```bash
uvicorn server_fastapi:app --reload
```

**What FastAPI gives us for free:**
1. **Routing:** `@app.get("/path")` instead of manual string parsing
2. **Type validation:** If you pass `"abc"` to `/users/{user_id}`, FastAPI returns a 422 error automatically
3. **JSON serialization:** Return a Python dict, get JSON automatically
4. **Auto-documentation:** Visit `http://localhost:8000/docs` for an interactive API explorer
5. **Async support:** Built-in support for async/await

### Exercise 2.2: Understanding Type Hints

FastAPI uses Python type hints for validation. Try this:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/multiply")
def multiply(a: int, b: int):
    return {"result": a * b}
```

**Test:**
- `http://localhost:8000/multiply?a=5&b=3` → `{"result": 15}`
- `http://localhost:8000/multiply?a=hello&b=3` → Error 422

**Why this matters:** Without type hints, you'd have to manually check:
```python
# Without FastAPI (what we'd have to write ourselves)
def multiply():
    a = request.args.get('a')
    b = request.args.get('b')
    if a is None or b is None:
        return error("Missing parameters")
    try:
        a = int(a)
        b = int(b)
    except ValueError:
        return error("Parameters must be integers")
    return {"result": a * b}
```

FastAPI saves us from writing all that boilerplate.

---

## Chapter 3: What is JSON? (Really)

JSON is the language of web APIs. But what IS it?

### Exercise 3.1: JSON is Just Text

```python
import json

# Python dictionary
data = {
    "name": "Alice",
    "age": 30,
    "is_student": False,
    "courses": ["Math", "Science"],
    "address": {
        "city": "New York",
        "zip": "10001"
    }
}

# Convert to JSON string
json_string = json.dumps(data, indent=2)
print(json_string)

# Convert back to Python
parsed = json.loads(json_string)
print(parsed["name"])  # Alice
```

**Key insight:** JSON is a text format that looks like Python dicts. Every programming language can read it.

**JSON rules:**
- Keys MUST be strings with double quotes: `"key"` not `'key'`
- Values can be: strings, numbers, booleans, null, arrays, objects
- No comments allowed
- No trailing commas

### Exercise 3.2: Reading JSON from a File

```python
import json

# Save API response to file
response = {
    "job_id": "abc123",
    "status": "completed",
    "model_url": "/outputs/abc123/model.glb"
}

with open("response.json", "w") as f:
    json.dump(response, f, indent=2)

# Read it back
with open("response.json", "r") as f:
    loaded = json.load(f)
    print(loaded["status"])
```

**This is how our app stores and retrieves job status.**

---

## Chapter 4: File Uploads (How Photos Get to the Server)

### Exercise 4.1: Receiving a File

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save the uploaded file
    contents = await file.read()
    with open(f"uploaded_{file.filename}", "wb") as f:
        f.write(contents)
    
    return {
        "filename": file.filename,
        "size": len(contents),
        "content_type": file.content_type
    }

@app.get("/", response_class=HTMLResponse)
def upload_form():
    return """
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <button type="submit">Upload</button>
    </form>
    """
```

**How this works:**
1. Browser renders a form with a file input
2. User selects a file and clicks Upload
3. Browser sends a `multipart/form-data` POST request
4. FastAPI parses it and gives us an `UploadFile` object
5. We read the bytes and save to disk

**The multipart format:**
```
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="file"; filename="photo.jpg"
Content-Type: image/jpeg

[binary image data here]
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

**Why multipart?** HTTP can only send text. Binary files (images, videos) need special encoding. Multipart wraps binary data in text boundaries.

---

## Chapter 5: Background Tasks (Why the Server Doesn't Hang)

### The Problem

If AI takes 60 seconds, and we do it in the request handler:
```python
@app.post("/generate")
def generate(image: UploadFile):
    result = run_ai_for_60_seconds(image)  # Browser waits 60 seconds!
    return result
```

The browser shows a spinning loader for 60 seconds. If the user refreshes, the request cancels. If the server restarts, the work is lost.

### Exercise 5.1: The Wrong Way (Blocking)

```python
import time
from fastapi import FastAPI

app = FastAPI()

def slow_work():
    print("Starting slow work...")
    time.sleep(10)  # Pretend this is AI taking 10 seconds
    print("Slow work done!")
    return "result"

@app.post("/blocking")
def blocking_endpoint():
    result = slow_work()  # The request waits here
    return {"result": result}
```

**Test:** Visit `/blocking` in your browser. The page loads for 10 seconds. During that time, the server can't handle other requests.

### Exercise 5.2: The Right Way (Background Task)

```python
import time
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

def slow_work(job_id: str):
    print(f"[{job_id}] Starting slow work...")
    time.sleep(10)
    print(f"[{job_id}] Done!")

@app.post("/background")
def background_endpoint(background_tasks: BackgroundTasks):
    job_id = "job_123"
    background_tasks.add_task(slow_work, job_id)
    return {"job_id": job_id, "status": "started"}
```

**Test:** Visit `/background`. The response comes back INSTANTLY: `{"job_id": "job_123", "status": "started"}`

The slow work happens AFTER the response is sent.

**But there's a problem:** How does the user know when it's done?

**Answer:** They need a way to check status. That's why we have job IDs and polling.

---

## Chapter 6: Job Queues (The Simplest Possible Version)

A queue is just a line. First in, first out.

### Exercise 6.1: Build a Queue from Scratch

```python
from dataclasses import dataclass
from enum import Enum
import time
import uuid

class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

@dataclass
class Job:
    id: str
    status: Status
    created_at: float
    result: str = None

# Our "database" is just a dictionary in memory
jobs = {}

def create_job() -> Job:
    job = Job(
        id=str(uuid.uuid4())[:8],
        status=Status.PENDING,
        created_at=time.time()
    )
    jobs[job.id] = job
    return job

def start_job(job_id: str):
    if job_id in jobs:
        jobs[job.id].status = Status.RUNNING
        print(f"Job {job_id} is now RUNNING")

def complete_job(job_id: str, result: str):
    if job_id in jobs:
        jobs[job.id].status = Status.DONE
        jobs[job.id].result = result
        print(f"Job {job_id} is DONE")

def get_job(job_id: str) -> Job:
    return jobs.get(job_id)

# Test it
job = create_job()
print(f"Created job: {job.id}")

start_job(job.id)

# Simulate work
import threading
def do_work(job_id):
    time.sleep(2)
    complete_job(job_id, "mesh.glb")

thread = threading.Thread(target=do_work, args=(job.id,))
thread.start()

# Poll for status
while True:
    j = get_job(job.id)
    print(f"Status: {j.status.value}")
    if j.status in (Status.DONE, Status.FAILED):
        print(f"Result: {j.result}")
        break
    time.sleep(0.5)
```

**What this teaches:**
1. Jobs have states (pending → running → done)
2. We store them in memory (dictionary)
3. We use threads so work doesn't block
4. We poll to check status

**Our real app does exactly this,** but with:
- `asyncio` instead of threads
- File I/O (saving images and meshes)
- More states (processing_coarse, coarse_ready, refining)

---

## Chapter 7: Static Files (How Frontend Code Gets to the Browser)

When you open our app, the browser needs:
- `index.html` (the page structure)
- `style.css` (the colors and layout)
- `app.js` (the JavaScript logic)
- Three.js (from a CDN)

### Exercise 7.1: Serving Static Files

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Create a "public" folder
os.makedirs("public", exist_ok=True)

# Write some files
with open("public/index.html", "w") as f:
    f.write("<html><body><h1>Hello</h1><link rel='stylesheet' href='/static/style.css'></body></html>")

with open("public/style.css", "w") as f:
    f.write("body { background: #222; color: white; }")

# Mount the static files at /static
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
def root():
    return FileResponse("public/index.html")
```

**What `mount` means:** Any request starting with `/static/` will look in the `public/` directory.

- `GET /static/style.css` → reads `public/style.css`
- `GET /static/app.js` → reads `public/app.js`

**Why separate static from dynamic?**
- Static files: same for every user (CSS, JS)
- Dynamic responses: different for every user (job status, generated meshes)

---

## Chapter 8: Environment Variables (Configuration Without Code Changes)

Environment variables are how we configure the app without editing code.

### Exercise 8.1: Reading Environment Variables

```python
import os

# Set an environment variable (in your terminal before running)
# export MY_APP_PORT=8000

port = os.getenv("MY_APP_PORT", "3000")  # Default to 3000 if not set
print(f"Port: {port}")

debug = os.getenv("DEBUG", "false").lower() == "true"
print(f"Debug mode: {debug}")
```

**Why environment variables?**
1. **Same code, different configs:** Local dev uses port 8000. Production uses port 80.
2. **Secrets:** API keys, database passwords. Never hardcode them.
3. **Feature flags:** `USE_INSTANTMESH=true` enables AI. `USE_INSTANTMESH=false` uses mock.

**Our app's environment variables:**
```bash
export DEVICE=cuda           # Which GPU to use
export USE_INSTANTMESH=true  # Enable real AI
export FP16=true            # Use half precision
export MAX_CONCURRENT_JOBS=2 # Parallel jobs
```

---

## Chapter 9: The Complete Request Lifecycle (Putting It All Together)

Let's trace what happens when a user uploads a photo:

```
1. BROWSER: User clicks "Upload"
   → JavaScript reads the file
   → Creates FormData object
   → fetch('/generate-3d', {method: 'POST', body: formData})

2. NETWORK: HTTP POST request travels over the internet
   → Headers: Content-Type: multipart/form-data
   → Body: binary image data

3. FASTAPI: Receives the request
   → Matches route @app.post("/generate-3d")
   → Parses multipart form
   → Validates file type (JPEG/PNG/WebP)
   → Creates UploadFile object

4. OUR CODE: Saves the file
   → Generates job ID (uuid4)
   → Creates upload directory
   → Writes image bytes to disk
   → Creates job in queue (status: PENDING)

5. BACKGROUND: Starts background task
   → Returns immediately to browser: {job_id: "abc123", status: "pending"}
   → Browser shows "Uploading..."

6. AI PIPELINE: (happens after response)
   → Load image from disk
   → Remove background (rembg)
   → Generate 6 views (Zero123++ diffusion)
   → Reconstruct 3D (LRM)
   → Extract mesh (FlexiCubes)
   → Save as GLB
   → Update job status: COARSE_READY

7. REFINEMENT:
   → Load coarse mesh
   → Subdivide (add triangles)
   → Smooth (Taubin)
   → Ensure UVs
   → Save as final.glb
   → Update job status: COMPLETED

8. BROWSER: Polls every second
   → GET /jobs/abc123
   → Status: pending → processing → coarse_ready → refining → completed
   → When coarse_ready: loads preview.glb in Three.js viewer
   → When completed: loads final.glb

9. USER: Sees 3D model, rotates it, downloads it
```

**This entire pipeline is what our app does.** Every step is necessary. Every step can fail. Every step needs to be understood.

---

## Exercises for This Part

### Exercise A: Build a To-Do API
Create a FastAPI app with:
- `POST /todos` - Create a todo (title, description)
- `GET /todos` - List all todos
- `GET /todos/{id}` - Get one todo
- `PUT /todos/{id}` - Update a todo
- `DELETE /todos/{id}` - Delete a todo

Store todos in a dictionary (like our job queue).

### Exercise B: File Upload with Progress
Create an endpoint that:
1. Accepts a file upload
2. Saves it in chunks (don't load entire file into memory)
3. Returns the file size and SHA256 hash

### Exercise C: Background Job System
Create a system where:
1. User submits a "job" (just a number to calculate factorial of)
2. Server returns job ID immediately
3. Background task calculates factorial (simulate slowness with sleep)
4. User polls `/jobs/{id}` for status
5. When done, result is available

### Exercise D: Environment-Aware Config
Create a config system that:
1. Reads `.env` file for local development
2. Falls back to environment variables
3. Has sensible defaults
4. Validates required variables on startup

---

**Next:** Part B: Async, Concurrency, and Polling — with runnable Python exercises.
