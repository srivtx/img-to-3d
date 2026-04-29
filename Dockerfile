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

# Install Hugging Face hub for model downloads
RUN pip install --no-cache-dir huggingface-hub

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/

# Create directories
RUN mkdir -p models outputs

# Pre-download InstantMesh weights during build (optional - uncomment after testing)
# This makes the image ~4GB but speeds up first inference
# RUN python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('TencentARC/InstantMesh', local_dir='/app/models/instantmesh', local_files_only=False)"

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEVICE=cpu
ENV KEEP_MODELS_IN_MEMORY=true
ENV MAX_CONCURRENT_JOBS=1

# Run the app
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
