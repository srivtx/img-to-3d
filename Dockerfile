FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/

# Create directories
RUN mkdir -p models outputs

# ----- Optional: enable real InstantMesh inference -----
# Default HF Space tier is CPU-only and InstantMesh requires CUDA + nvdiffrast,
# so we ship the image with USE_INSTANTMESH off (mock icosphere). To enable
# real inference on a GPU-enabled Space:
#
# 1. Switch the Space hardware to a GPU tier.
# 2. Uncomment the block below to clone the InstantMesh repo, install its
#    pinned deps, and pre-download weights into the image.
# 3. Set USE_INSTANTMESH=true and DEVICE=cuda below.
#
# RUN git clone https://github.com/TencentARC/InstantMesh.git /app/models/InstantMesh \
#  && pip install --no-cache-dir -r /app/models/InstantMesh/requirements.txt \
#  && pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast/ \
#  && python -c "from huggingface_hub import snapshot_download; \
#       snapshot_download('TencentARC/InstantMesh', \
#         local_dir='/app/models/instantmesh', \
#         local_dir_use_symlinks=False)"

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEVICE=cpu
ENV USE_INSTANTMESH=false
ENV KEEP_MODELS_IN_MEMORY=true
ENV MAX_CONCURRENT_JOBS=1

# Run the app
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
