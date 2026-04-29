#!/bin/bash
set -e

echo "=========================================="
echo "  InstantMesh Setup for Image-to-3D"
echo "=========================================="
echo ""

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# 1. Clone InstantMesh
echo "[1/3] Cloning InstantMesh repository..."
if [ ! -d "InstantMesh" ]; then
    git clone https://github.com/TencentARC/InstantMesh.git
    echo "✓ Cloned successfully"
else
    echo "✓ Already cloned, skipping"
fi

# 2. Install dependencies
echo ""
echo "[2/3] Installing InstantMesh dependencies..."
cd InstantMesh
pip install -r requirements.txt || {
    echo "⚠️  requirements.txt install had issues, trying core deps..."
    pip install diffusers transformers accelerate safetensors omegaconf einops rembg
}

# 3. Download weights
echo ""
echo "[3/3] Downloading model weights..."
echo ""
echo "Weights must be downloaded from HuggingFace."
echo "Run the following commands MANUALLY:"
echo ""
echo "  cd $MODELS_DIR/InstantMesh"
echo "  # Option A: Use huggingface-cli (install with: pip install huggingface-hub)"
echo "  huggingface-cli download TencentARC/InstantMesh --local-dir ./weights"
echo ""
echo "  # Option B: Manual download"
echo "  # Visit: https://huggingface.co/TencentARC/InstantMesh"
echo "  # Download all .safetensors / .bin files and config.json"
echo "  # Place them in: $MODELS_DIR/instantmesh/"
echo ""

mkdir -p "$MODELS_DIR/instantmesh"

echo "=========================================="
echo "  Setup Complete (Partial)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download weights (see commands above)"
echo "  2. Edit: app/pipeline/instantmesh.py"
echo "  3. Set USE_INSTANTMESH = True"
echo "  4. Implement _run_inference() with their actual API"
echo "  5. Restart server: python -m uvicorn app.main:app --reload"
echo ""
