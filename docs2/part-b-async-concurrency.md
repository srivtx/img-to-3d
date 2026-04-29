# Part B: Async, Concurrency, and Polling

## Chapter 1: The Blocking Problem

### Exercise 1.1: Prove Blocking Is Bad

```python
import time

def slow_task(name, seconds):
    print(f"[{name}] Starting...")
    time.sleep(seconds)
    print(f"[{name}] Done!")

start = time.time()
slow_task("A", 2)
slow_task("B", 2)
slow_task("C", 2)
print(f"Total: {time.time() - start:.1f}s")
# Output: ~6 seconds
```

**Why this matters:** If AI takes 60 seconds and we process sequentially, user B waits 60s before we even START their job.

### Exercise 1.2: Threading Helps (For I/O)

```python
import time
import threading

def slow_task(name, seconds):
    print(f"[{name}] Starting...")
    time.sleep(seconds)
    print(f"[{name}] Done!")

start = time.time()
threads = []
for name in ["A", "B", "C"]:
    t = threading.Thread(target=slow_task, args=(name, 2))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Total: {time.time() - start:.1f}s")
# Output: ~2 seconds!
```

**What happened:** While thread A sleeps, thread B runs. But Python has a GIL (Global Interpreter Lock) — only one thread runs Python code at a time. For I/O (sleep, network), threads help. For CPU-heavy work, they don't.

### Exercise 1.3: The GIL Problem

```python
import time
import threading

def cpu_task(name, n):
    total = 0
    for i in range(n):
        total += i ** 2
    print(f"[{name}] Done")

start = time.time()
t1 = threading.Thread(target=cpu_task, args=("A", 5_000_000))
t2 = threading.Thread(target=cpu_task, args=("B", 5_000_000))
t1.start()
t2.start()
t1.join()
t2.join()
print(f"Total: {time.time() - start:.1f}s")
# Output: Same as sequential! GIL prevents parallel CPU work.
```

**Lesson:** Threads don't speed up CPU work in Python. Use multiprocessing or accept that CPU work is serial.

---

## Chapter 2: Asyncio — Python's Solution

### Exercise 2.1: First Async Function

```python
import asyncio

async def say_hello():
    print("Hello...")
    await asyncio.sleep(1)  # Non-blocking!
    print("World!")

asyncio.run(say_hello())
```

**`async`**: This function can pause and resume.  
**`await`**: Pause here, let other tasks run.

### Exercise 2.2: Multiple Tasks Concurrently

```python
import asyncio
import time

async def task(name, seconds):
    print(f"[{name}] Starting")
    await asyncio.sleep(seconds)
    print(f"[{name}] Done")

async def main():
    start = time.time()
    await asyncio.gather(
        task("A", 2),
        task("B", 2),
        task("C", 2),
    )
    print(f"Total: {time.time() - start:.1f}s")
    # Output: ~2 seconds (not 6!)

asyncio.run(main())
```

**How it works:**
1. Task A starts, hits `await asyncio.sleep(2)`
2. Event loop pauses A, immediately starts B
3. Task B starts, hits `await asyncio.sleep(2)`
4. Event loop pauses B, immediately starts C
5. All three are "sleeping" but registered with the loop
6. After 2 seconds, all three wake up and finish

**One thread. No GIL issues. Thousands of concurrent I/O operations.**

### Exercise 2.3: Visualize the Event Loop

```python
import asyncio

async def worker(name, delay):
    for i in range(3):
        print(f"  [{name}] Step {i+1}")
        await asyncio.sleep(delay)
    print(f"  [{name}] FINISHED")

async def main():
    await asyncio.gather(
        worker("Fast", 0.3),
        worker("Medium", 0.5),
        worker("Slow", 0.7),
    )

asyncio.run(main())
```

**Watch the interleaving.** Fast finishes first. The loop constantly switches between tasks.

---

## Chapter 3: CPU Work in Async (The Trap)

### Exercise 3.1: CPU Work Blocks the Loop

```python
import asyncio
import time

def heavy_computation(n):
    """CPU-bound: no await, no I/O"""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

async def bad_endpoint():
    """This BLOCKS the entire server!"""
    result = heavy_computation(10_000_000)
    return {"result": result}

async def good_endpoint():
    """This runs CPU work in a thread, doesn't block."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, heavy_computation, 10_000_000)
    return {"result": result}
```

**Test:** Start the server. In two terminals:
```bash
# Terminal 1 - This will take a while
curl http://localhost:8000/bad-endpoint

# Terminal 2 - Run immediately after. This will WAIT until terminal 1 finishes!
curl http://localhost:8000/health
```

With `bad_endpoint`, the health check waits. With `good_endpoint`, it returns instantly.

**This is why our app uses:**
```python
await asyncio.to_thread(coarse_generator.generate, image_path, output_dir)
```

---

## Chapter 4: Polling — Build It From Scratch

### Exercise 4.1: Simple Polling Server

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()
jobs = {}

@app.post("/start")
async def start_job():
    import uuid
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "running", "progress": 0}
    
    # Start background work
    asyncio.create_task(do_work(job_id))
    return {"job_id": job_id}

async def do_work(job_id):
    for i in range(10):
        await asyncio.sleep(0.5)
        jobs[job_id]["progress"] = (i + 1) * 10
    jobs[job_id]["status"] = "done"

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    return jobs.get(job_id, {"error": "not found"})
```

### Exercise 4.2: Polling Client

```python
import requests
import time

# Start job
r = requests.post("http://localhost:8000/start")
job_id = r.json()["job_id"]
print(f"Job: {job_id}")

# Poll
while True:
    status = requests.get(f"http://localhost:8000/jobs/{job_id}").json()
    print(f"Progress: {status['progress']}%")
    if status["status"] == "done":
        break
    time.sleep(0.5)
```

### Exercise 4.3: Exponential Backoff

```python
def poll_smart(job_id):
    interval = 0.5
    while True:
        status = requests.get(f"http://localhost:8000/jobs/{job_id}").json()
        print(f"Progress: {status['progress']}%")
        if status["status"] == "done":
            return status
        time.sleep(interval)
        interval = min(interval * 1.5, 10)  # Cap at 10s
```

**Why backoff?** Early on, progress changes fast (we want responsive UI). Later, changes are slow (we save server resources).

---

## Chapter 5: Architecture Debate — Polling vs WebSocket vs SSE

### The Three Approaches

| Approach | How It Works | Best For |
|----------|-------------|----------|
| **Polling** | Browser asks every N seconds | Job status, simple setups |
| **WebSocket** | Persistent two-way connection | Chat, games, real-time |
| **SSE** | Persistent one-way (server → browser) | Live feeds, notifications |

### Debate: Polling (Our Choice)

**Pros:**
- Works everywhere (no special protocol)
- Stateless (any server can answer)
- Easy to debug (just HTTP requests)
- Works through firewalls/proxies
- Simple retry logic

**Cons:**
- Wasted requests (asking when nothing changed)
- Latency = up to poll interval
- More bandwidth
- Server load from repeated checks

**When to use:** Job processing (30-60s), file uploads, batch operations.

### Debate: WebSocket

**Pros:**
- True real-time (< 100ms latency)
- Server pushes only when data changes
- Bidirectional (browser can send too)
- Efficient after connection established

**Cons:**
- Complex connection management
- Need sticky sessions for scaling
- Firewalls may block
- Harder to debug
- Browser connection limits

**When to use:** Chat, collaborative editing, games, stock tickers.

### Debate: Server-Sent Events (SSE)

**Pros:**
- Simpler than WebSocket (HTTP-based)
- Automatic reconnection
- One-way is often all you need

**Cons:**
- One-way only (browser → server needs separate request)
- Connection limits per browser
- Not supported in all proxies

**When to use:** Live logs, progress bars, notifications.

### Our Decision

We chose **polling** because:
1. **Simplicity:** No connection state to manage
2. **Reliability:** Works on every network
3. **Scaling:** Any server can answer any poll
4. **Debugging:** Just curl commands
5. **Good enough:** 1-second updates for 30-60s jobs is fine

**Future evolution:** Switch to SSE for progress bars, keep polling as fallback.

---

**Next:** Part C — How AI Inference Actually Works (line-by-line deep dive)
