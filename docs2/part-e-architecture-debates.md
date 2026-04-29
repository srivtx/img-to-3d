# Part E: Architecture Debates — Why We Chose What We Chose

## Introduction

Every decision in our project was a trade-off. This document explores the alternatives we considered, the pros and cons of each, and why we made the choices we did.

We present these as **Decision Records** — a format used in software architecture to document why decisions were made.

---

## Debate 1: Mock Mode First vs. Real AI First

### Option A: Real AI First

**Approach:** Start by wiring InstantMesh. Don't build anything until the AI works.

**Pros:**
- You see real results immediately
- No wasted work on mock infrastructure
- Forces you to solve the hardest problem first

**Cons:**
- AI setup is complex (dependencies, GPU, weights)
- While debugging AI, you can't test the UI
- If AI takes 2 weeks to wire, you have nothing to show for 2 weeks
- The UI team is blocked

### Option B: Mock Mode First (Our Choice)

**Approach:** Build the entire app with a placeholder sphere. Wire AI later.

**Pros:**
- Immediate feedback — you see a working app on day 1
- UI/UX can be refined while AI is being wired
- API contract is established
- Each layer is independently testable
- If AI fails, you still have a working demo

**Cons:**
- You build infrastructure that might change when AI is wired
- Risk of over-engineering the mock path
- Team might get complacent ("it works, ship it")

### Our Decision

**We chose mock first.** The reasoning:

1. **Parallel workstreams:** Frontend team builds UI while backend team wires AI
2. **Psychological:** Seeing a sphere rotate in a browser is motivating. Debugging CUDA errors for 3 days is demotivating.
3. **Testing:** We can test uploads, progress bars, downloads, and 3D rendering without waiting 30 seconds per test
4. **Fallback:** If we present to investors and the AI fails, we still have a smooth demo

**The sphere is a promise.** It says "imagine this, but accurate."

---

## Debate 2: FastAPI vs. Flask vs. Django

### Option A: Flask

**Pros:**
- Simple, minimal, easy to learn
- Huge community
- Flexible (you add what you need)

**Cons:**
- Not async by default (though Flask 2+ supports it)
- No automatic API documentation
- Manual request validation
- Slower for I/O-bound workloads

### Option B: Django

**Pros:**
- Batteries included (ORM, admin, auth)
- Mature, well-documented
- Great for traditional web apps

**Cons:**
- Heavyweight for an API-only service
- Not designed for async (Django 3.1+ has limited async)
- ORM is overkill for our use case
- Opinionated structure might fight our pipeline architecture

### Option C: FastAPI (Our Choice)

**Pros:**
- Native async/await support
- Automatic request validation via type hints
- Auto-generated OpenAPI documentation
- Performance (on par with Node.js/Go for I/O-bound)
- Modern Python features

**Cons:**
- Newer ecosystem than Flask/Django
- Some libraries assume Flask patterns
- Learning curve if you're used to sync frameworks

### Our Decision

**FastAPI.** Because:
1. **Async is essential:** Our server handles uploads while AI runs in background. Async prevents blocking.
2. **Type validation:** `image: UploadFile = File(...)` automatically validates file uploads.
3. **Documentation:** We get an interactive API explorer for free at `/docs`.
4. **Performance:** Benchmarks show FastAPI handling 10,000+ concurrent connections.

**When to use Flask instead:** Simple APIs, team already knows Flask, need specific Flask extensions.

**When to use Django instead:** Full web app with database, admin panel, user authentication.

---

## Debate 3: Polling vs. WebSockets vs. Server-Sent Events

### Option A: Polling (Our Choice)

**How it works:** Browser asks "status?" every N seconds.

**Pros:**
- Works everywhere (just HTTP)
- Stateless (any server can answer)
- Easy to debug (just curl)
- Works through all firewalls/proxies
- Simple retry logic

**Cons:**
- Wasted requests (asking when nothing changed)
- Latency = up to poll interval
- More bandwidth
- Server load scales with number of clients

### Option B: WebSockets

**How it works:** Persistent two-way connection. Server pushes updates.

**Pros:**
- True real-time (< 100ms latency)
- Server only sends when data changes
- Bidirectional (client can send too)

**Cons:**
- Complex connection management (reconnects, heartbeats)
- Need sticky sessions for load balancing
- Firewalls may block WebSocket port (not 80/443)
- Harder to debug (can't just curl)
- Browser connection limits (~6 per domain)

### Option C: Server-Sent Events (SSE)

**How it works:** Persistent one-way connection. Server streams events.

**Pros:**
- Simpler than WebSockets (HTTP-based)
- Automatic reconnection built into browser
- One-way is often all you need (status updates)

**Cons:**
- One-way only (need separate request for client → server)
- 6 connection limit per browser
- Not supported by all HTTP proxies

### Our Decision

**Polling with exponential backoff.** Because:
1. **Simplicity:** No connection state, no reconnection logic
2. **Reliability:** Works on every network, every browser
3. **Scaling:** Any server can answer any poll (stateless)
4. **Good enough:** For 30-60 second jobs, 1-second latency is fine

**Future evolution:** SSE for the progress bar (smoother updates), polling as fallback for older browsers.

**When to use WebSockets:** Chat apps, multiplayer games, collaborative editing, stock tickers.

---

## Debate 4: In-Memory Queue vs. Redis vs. Celery

### Option A: In-Memory Queue (Our Choice)

**How it works:** Python dictionary + asyncio locks.

**Pros:**
- Zero setup (no Redis server to install)
- Fast (no network calls)
- Simple to understand and debug
- Perfect for single-server deployments

**Cons:**
- Jobs lost on server restart
- Can't scale to multiple servers
- Memory usage grows with queue size
- No persistence

### Option B: Redis Queue

**How it works:** Redis stores job state. Multiple workers read from Redis.

**Pros:**
- Persistent (survives server restarts)
- Scales to multiple servers
- Can inspect queue from CLI
- Battle-tested (used by millions of apps)

**Cons:**
- Need to run Redis server
- Network overhead per operation
- More complex setup
- Another service to monitor

### Option C: Celery + Redis/RabbitMQ

**How it works:** Celery is a distributed task queue. Redis/RabbitMQ is the broker.

**Pros:**
- Mature, feature-rich
- Retries, scheduling, rate limiting
- Monitoring dashboard (Flower)
- Language agnostic

**Cons:**
- Heavyweight (Redis + Celery worker + Flower)
- Complex configuration
- Overkill for simple use cases
- Learning curve

### Our Decision

**In-memory queue.** Because:
1. **MVP first:** We're validating the product, not scaling to millions of users
2. **Speed of development:** Zero infrastructure setup
3. **Understandability:** A dictionary is easier to debug than a distributed system
4. **Migration path:** When we need to scale, swapping in Redis is straightforward

**Migration plan:**
1. Phase 1: In-memory (now)
2. Phase 2: Redis for persistence
3. Phase 3: Celery for distributed workers

**When to use Redis immediately:** Multiple servers, need persistence, or team already uses Redis.

---

## Debate 5: Local GPU vs. Colab vs. Cloud GPU

### Option A: Local GPU (Mac MPS / NVIDIA)

**Pros:**
- No ongoing costs
- Fast iteration (no uploading/downloading)
- Full control over environment
- Works offline

**Cons:**
- Mac MPS has compatibility issues (some ops not supported)
- Consumer GPUs have limited VRAM (8-24 GB)
- Your machine is tied up during training/inference
- Hard to share with team

### Option B: Google Colab Free T4 (Our Choice for Testing)

**Pros:**
- Free
- 16 GB VRAM T4 GPU
- Pre-installed ML libraries
- Easy to share (notebook link)

**Cons:**
- Session disconnects after ~12 hours or idle timeout
- No persistent storage (must re-download weights each session)
- No guaranteed GPU availability
- Can't run as a persistent service

### Option C: Hugging Face Spaces (Our Choice for Deployment)

**Pros:**
- Free CPU tier
- Paid GPU upgrades (L4, A10G)
- Persistent URL
- Built for ML demos
- Easy Git integration

**Cons:**
- Sleeps after inactivity (free tier)
- GPU costs add up ($0.50-3/hour)
- Limited to port 7860
- No WebSocket support

### Option D: RunPod / Vast.ai

**Pros:**
- Cheap GPU rental ($0.20-0.50/hour for RTX 3090)
- Persistent (runs until you stop it)
- Full control (root access)
- Can run any software

**Cons:**
- Must manage infrastructure yourself
- No persistent storage (unless you pay extra)
- Must set up tunnels for public access
- Costs money

### Our Decision

**Local development → Colab for testing → HF Spaces for demo**

**Reasoning:**
1. **Develop locally** (Mac MPS): Fast iteration, edit code, see changes instantly
2. **Test on Colab T4** (free): Verify it works on the actual GPU users will have
3. **Deploy to HF Spaces** (credits): Persistent URL for sharing

**Cost optimization:**
- Development: $0 (local)
- Testing: $0 (Colab free tier)
- Demo: ~$5-10/month (HF Spaces CPU + occasional GPU upgrades)

---

## Debate 6: GLB vs. OBJ vs. PLY vs. Gaussian Splatting

### Option A: GLB (Our Choice)

**Pros:**
- Single file (mesh + materials + textures)
- Native browser support (Three.js loads directly)
- Compact binary format
- Standard (Khronos Group)

**Cons:**
- Binary (not human-readable)
- Less supported in CAD software than STEP/IGES

### Option B: OBJ

**Pros:**
- Human-readable text format
- Universal support (every 3D app opens it)
- Simple structure

**Cons:**
- Separate files for textures (.mtl, .png)
- Larger file size
- No animation support

### Option C: PLY

**Pros:**
- Simple format
- Good for point clouds
- Stores per-vertex data easily

**Cons:**
- Not standard for web
- Larger than GLB
- No material support

### Option D: Gaussian Splatting (.ply/.splat)

**Pros:**
- Photorealistic rendering
- Fast (millions of gaussians render in real-time)
- No mesh topology issues

**Cons:**
- Hard to edit (no explicit surfaces)
- Large file sizes
- Not universally supported yet
- Hard to animate

### Our Decision

**GLB for mesh output.** Because:
1. **Web-first:** Our app runs in browsers. GLB is the web standard.
2. **Single file:** Users download one file, not a folder.
3. **Tools:** Blender, Unity, Unreal all import GLB.

**Future:** Offer Gaussian Splatting as an alternative output format for users who want photorealistic rendering.

---

## Debate 7: Vertex Colors vs. Textures

### Option A: Vertex Colors (Our Choice for Preview)

**How it works:** Each vertex stores an RGB color. GPU interpolates between vertices.

**Pros:**
- Simple (no UV mapping needed)
- Single file (no separate texture image)
- Good enough for coarse preview
- Fast to generate

**Cons:**
- Limited detail (color blends between vertices)
- Large meshes needed for sharp color transitions
- Not standard for production assets

### Option B: UV Textures

**How it works:** A 2D image (texture) is mapped onto the mesh using UV coordinates.

**Pros:**
- High detail (1024×1024 texture = 1 million pixels of detail)
- Standard for production
- Can use PBR materials (roughness, metalness maps)

**Cons:**
- Need UV unwrapping (complex for arbitrary meshes)
- Separate texture files
- More compute to generate

### Our Decision

**Vertex colors for preview, textures for final (future).**

**Reasoning:**
1. **Speed:** Vertex colors are extracted directly from the triplane. No additional generation step.
2. **Simplicity:** No UV mapping needed for the coarse preview.
3. **Good enough:** For a 30-second preview, vertex colors show the shape and approximate colors.

**Future work:** Project the original photo onto the mesh to create a texture. This gives photo-realistic detail.

---

## Debate 8: FP16 vs. FP32

### Option A: FP32 (Full Precision)

**Pros:**
- Maximum accuracy
- No numerical issues
- Easier debugging

**Cons:**
- 2x memory usage
- 2x slower on some GPUs
- Larger checkpoint files

### Option B: FP16 (Half Precision, Our Choice)

**Pros:**
- 2x less VRAM
- 2x faster memory bandwidth
- Supported by modern GPUs (Tensor Cores)
- Usually no noticeable quality loss

**Cons:**
- Can overflow/underflow (numbers too big/small)
- Some operations unstable in FP16
- Not supported on older GPUs

### Our Decision

**FP16 on CUDA, FP32 on CPU/MPS.**

**Reasoning:**
1. **VRAM is tight:** T4 has 16 GB. FP16 lets us fit larger models.
2. **Speed:** Tensor Cores on NVIDIA GPUs are optimized for FP16.
3. **Safety:** We fall back to FP32 on CPU (no Tensor Cores) and MPS (FP16 support is limited).

**Code:**
```python
torch_dtype = torch.float16 if (FP16 and device.type == "cuda") else torch.float32
```

---

## Debate 9: InstantMesh vs. CRM vs. TripoSR

### InstantMesh (Our Choice)

**Pros:**
- Open source (Apache 2.0)
- Good quality/speed trade-off
- Well-documented
- Active community

**Cons:**
- VRAM hungry (~12 GB peak)
- Occasional artifacts
- Setup complexity (shims, custom pipeline)

### CRM

**Pros:**
- Faster (~10-20s)
- High fidelity
- Direct mesh prediction (no multi-view needed)

**Cons:**
- Newer, less mature
- Smaller community
- Unknown long-term support

### TripoSR

**Pros:**
- Very fast (~5-15s)
- Excellent quality
- Stability AI backing

**Cons:**
- License restrictions (research vs. commercial)
- Weights not freely available
- Less transparent

### Our Decision

**InstantMesh.** Because:
1. **Open source:** We can inspect, modify, and distribute.
2. **Proven:** Thousands of users, works on standard hardware.
3. **Free:** No per-generation cost.

**Future:** Add CRM as an alternative for users who want speed over compatibility.

---

## Debate 10: Monolithic vs. Microservices

### Option A: Monolith (Our Choice)

**How it works:** One Python process handles everything — API, queue, AI inference.

**Pros:**
- Simple to develop and deploy
- No network overhead between components
- Easy to debug (one log file)
- Fast iteration

**Cons:**
- Can't scale API and AI independently
- If AI crashes, API goes down too
- Limited to one machine's GPU

### Option B: Microservices

**How it works:**
- API service (FastAPI) → handles HTTP requests
- Queue service (Redis) → stores job state
- Worker service (GPU) → runs AI inference
- Frontend service (nginx) → serves static files

**Pros:**
- Scale independently (10 API servers, 2 GPU workers)
- Different tech stacks per service
- Fault isolation (AI crash doesn't kill API)
- Team autonomy

**Cons:**
- Network latency between services
- Complex deployment
- Harder to debug (distributed logs)
- Overhead (Kubernetes, service mesh)

### Our Decision

**Monolith for now, microservices later.**

**Reasoning:**
1. **Team size:** 1-2 developers. Microservices need more people.
2. **Complexity budget:** We're already dealing with ML integration. Don't add distributed systems complexity.
3. **Premature optimization:** We don't have scaling problems yet.

**Migration path:**
1. Phase 1: Monolith (now)
2. Phase 2: Extract AI worker to separate process (same machine)
3. Phase 3: Extract AI worker to separate service (different machine)
4. Phase 4: Kubernetes with auto-scaling

---

## Summary of All Decisions

| Decision | Chose | Because |
|----------|-------|---------|
| **Mock first** | Yes | Parallel workstreams, instant feedback |
| **Framework** | FastAPI | Async, type validation, auto-docs |
| **Polling** | Yes | Simple, reliable, stateless |
| **Queue** | In-memory | Zero setup, fast, MVP |
| **GPU platform** | Colab + HF Spaces | Free testing, paid deployment |
| **Output format** | GLB | Web standard, single file |
| **Colors** | Vertex colors | Fast, simple, good enough |
| **Precision** | FP16 (CUDA) | VRAM efficient, fast |
| **AI model** | InstantMesh | Open source, proven |
| **Architecture** | Monolith | Team size, simplicity |

---

## When to Revisit Each Decision

| Decision | Revisit When... |
|----------|----------------|
| Mock mode | AI is wired and stable |
| In-memory queue | >100 concurrent users |
| Polling | Need <100ms updates |
| Colab | Need persistent 24/7 service |
| Vertex colors | Users demand texture quality |
| Monolith | Team grows to 5+ developers |
| InstantMesh | Better open-source model emerges |

---

**Next:** Part F — Advanced Topics
