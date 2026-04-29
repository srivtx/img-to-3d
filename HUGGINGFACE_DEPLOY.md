# Deploy to Hugging Face Spaces

This guide walks you through deploying the Image-to-3D app on **Hugging Face Spaces** using your credits.

---

## Why Hugging Face Spaces?

| Feature | Benefit |
|---------|---------|
| **Free CPU tier** | Test forever at $0 |
| **GPU upgrades** | Use your credits for A10G/L4/A100 |
| **Persistent storage** | Models stay downloaded |
| **Public URL** | Share instantly |
| **Zero devops** | No server management |

---

## Method 1: Deploy via Web UI (Easiest)

### Step 1: Create Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in:
   - **Owner:** Your username
   - **Space name:** `image-to-3d` (or whatever)
   - **License:** MIT
   - **Space SDK:** Select **Docker**
   - **Space hardware:** Start with **CPU basic** (free), upgrade later
4. Click **"Create Space"**

### Step 2: Upload Files

In your new Space, go to **Files** → **Upload files**, upload these from your local project:

```
app/
frontend/
requirements.txt
Dockerfile
```

Or use Git:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/image-to-3d
cd image-to-3d
# Copy your project files here
git add .
git commit -m "Initial deployment"
git push
```

### Step 3: Wait for Build

HF Spaces will automatically build the Docker image. Check the **Logs** tab.

### Step 4: Test

Once build completes, click **"App"** tab. You'll see your upload UI!

---

## Method 2: Deploy via CLI

```bash
# 1. Install HF CLI
pip install huggingface-hub

# 2. Login
huggingface-cli login

# 3. Create space (Docker, private initially)
huggingface-cli repo create image-to-3d \
  --type space \
  --sdk docker \
  --private

# 4. Clone and push
git clone https://huggingface.co/spaces/YOUR_USERNAME/image-to-3d
cd image-to-3d

# Copy all project files
rsync -av ../image-to-3d/app/ ./app/
rsync -av ../image-to-3d/frontend/ ./frontend/
cp ../image-to-3d/requirements.txt .
cp ../image-to-3d/Dockerfile .

git add .
git commit -m "Deploy Image-to-3D"
git push
```

---

## Upgrading to GPU

Once deployed and working on CPU:

1. Go to Space **Settings**
2. Find **"Space Hardware"** section
3. Select:
   - **Nvidia L4** (~$0.50/hour) — great for InstantMesh
   - **Nvidia A10G** (~$1.00/hour) — faster
   - **Nvidia A100** (~$3-4/hour) — overkill for this
4. Click **Upgrade**

The Space will restart with GPU. Update the `Dockerfile` environment:

```dockerfile
ENV DEVICE=cuda
ENV FP16=true
```

---

## Setting Up InstantMesh on HF Spaces

### Option A: Download Weights at Runtime (Slower first start)

Add to your Space's `README.md` or create a startup script. The first user will wait ~2-3 minutes for downloads.

### Option B: Pre-download Weights into the Docker Image (Faster)

Add to `Dockerfile`:

```dockerfile
# Pre-download InstantMesh weights during build
RUN pip install huggingface-hub
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('TencentARC/InstantMesh', local_dir='/app/models/instantmesh')"
```

**Note:** This makes the Docker image ~4GB. HF has storage limits, but it's usually fine.

### Option C: Use Persistent Storage (Recommended)

HF Spaces has persistent storage at `/data`. Download weights there once:

```bash
# In your Space's terminal (Factory reboot tab)
huggingface-cli download TencentARC/InstantMesh --local-dir /data/models/instantmesh
```

Then update `app/pipeline/instantmesh.py` to look in `/data/models/`.

---

## Important: HF Spaces Limitations

| Limitation | Workaround |
|------------|------------|
| **Sleep after inactivity** | Free CPU sleeps after ~48hrs idle. GPU sleeps after ~1hr. Use **"Keep awake"** in settings or ping it periodically |
| **No WebSocket** | Polling works fine, already implemented |
| **Port 7860 only** | Already configured in Dockerfile |
| **Storage limits** | Use `/data` for large models, or download at runtime |

---

## Environment Variables on HF

Set these in Space **Settings** → **Variables and Secrets**:

| Variable | Value | Description |
|----------|-------|-------------|
| `DEVICE` | `cuda` (GPU) or `cpu` (free tier) | PyTorch device |
| `USE_INSTANTMESH` | `true` or `false` | Enable real inference |
| `MAX_CONCURRENT_JOBS` | `1` | HF Spaces works best with 1 job |

---

## Cost Estimate

| Tier | Cost | Usage |
|------|------|-------|
| **CPU Basic** | $0/month | Testing, development |
| **Nvidia L4** | ~$0.50/hr | Real inference, ~2-5s generation |
| **Nvidia A10G** | ~$1.00/hr | Faster inference |

With $10-20 in credits, you can run GPU for 10-40 hours — plenty for testing and demos.

---

## Next Steps After Deploy

1. **Test upload** on the public URL
2. **Upgrade to GPU** when ready for real models
3. **Download InstantMesh weights** (via terminal or pre-build)
4. **Enable USE_INSTANTMESH** and restart
5. **Share the URL** — anyone can use it!

---

## Troubleshooting

### "Build failed"
- Check **Logs** tab for error details
- Common issue: `requirements.txt` has a package that fails to install
- Fix: Pin versions or remove problematic packages

### "App sleeps too quickly"
- Upgrade to **Pro** ($9/month) for persistent CPU
- Or use a cron job to ping your Space every hour
- GPU always sleeps after inactivity (expected)

### "Out of memory"
- Reduce `MAX_CONCURRENT_JOBS` to `1`
- Use smaller models or FP16
- Upgrade to GPU with more VRAM

---

## Quick Reference

```bash
# Your Space URL will be:
https://huggingface.co/spaces/YOUR_USERNAME/image-to-3d

# The app will be accessible at:
https://YOUR_USERNAME-image-to-3d.hf.space
```
