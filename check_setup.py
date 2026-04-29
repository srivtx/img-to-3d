"""Check if InstantMesh is properly set up and ready for inference."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
INSTANTMESH_CODE = MODELS_DIR / "InstantMesh"
INSTANTMESH_WEIGHTS = MODELS_DIR / "instantmesh"

def check():
    print("=" * 50)
    print("  Image-to-3D Setup Validator")
    print("=" * 50)
    print()

    ok = True

    # 1. Check code directory
    if INSTANTMESH_CODE.exists():
        print(f"✓ InstantMesh code found at: {INSTANTMESH_CODE}")
    else:
        print(f"✗ InstantMesh code NOT found.")
        print(f"  Expected: {INSTANTMESH_CODE}")
        print(f"  Run: ./setup_instantmesh.sh")
        ok = False

    # 2. Check weights directory
    if INSTANTMESH_WEIGHTS.exists():
        files = list(INSTANTMESH_WEIGHTS.iterdir())
        if files:
            print(f"✓ Weights directory has {len(files)} files")
            for f in files[:5]:
                print(f"    - {f.name}")
            if len(files) > 5:
                print(f"    ... and {len(files)-5} more")
        else:
            print(f"⚠ Weights directory exists but is EMPTY")
            print(f"  Download weights from: https://huggingface.co/TencentARC/InstantMesh")
            ok = False
    else:
        print(f"✗ Weights directory NOT found.")
        print(f"  Expected: {INSTANTMESH_WEIGHTS}")
        ok = False

    # 3. Check Python imports
    print()
    print("Checking Python imports...")
    sys.path.insert(0, str(INSTANTMESH_CODE))
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} (device: {torch.device('cpu')})")
    except ImportError:
        print("✗ PyTorch not installed")
        ok = False

    try:
        import diffusers
        print(f"✓ Diffusers {diffusers.__version__}")
    except ImportError:
        print("✗ Diffusers not installed (pip install diffusers)")
        ok = False

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed (pip install transformers)")
        ok = False

    # 4. Check config flag
    print()
    from app.pipeline.instantmesh import USE_INSTANTMESH
    if USE_INSTANTMESH:
        print("✓ USE_INSTANTMESH = True in app/pipeline/instantmesh.py")
    else:
        print("⚠ USE_INSTANTMESH = False")
        print("  Edit app/pipeline/instantmesh.py and set USE_INSTANTMESH = True")

    print()
    print("=" * 50)
    if ok and USE_INSTANTMESH:
        print("  ✓ READY for real inference!")
    elif ok and not USE_INSTANTMESH:
        print("  ⚠ Almost ready — enable USE_INSTANTMESH flag")
    else:
        print("  ✗ NOT ready — fix issues above")
    print("=" * 50)

    return ok

if __name__ == "__main__":
    check()
