"""
InstantMesh coarse generation wrapper.

This file handles loading and running the InstantMesh model for
image-to-3D generation. It auto-detects whether InstantMesh is set up
and falls back to mock mode (icosphere) if not.

Environment variables:
    USE_INSTANTMESH: "true" to enable real inference
    DEVICE: "cuda", "mps", or "cpu"
"""

import os
import sys
import warnings
import traceback
from pathlib import Path
from typing import Tuple, Optional

import torch
import numpy as np
from PIL import Image

from app.core.config import MODELS_DIR, DEVICE, FP16

# ============================================================================
# CONFIGURATION
# ============================================================================
USE_INSTANTMESH = os.getenv("USE_INSTANTMESH", "false").lower() == "true"

# Path to InstantMesh code (relative to project root)
INSTANTMESH_CODE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "InstantMesh"
INSTANTMESH_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "instantmesh"

# ============================================================================
# Try to import InstantMesh modules
# ============================================================================
_HAS_INSTANTMESH = False
_INSTANTMESH_ERROR = None

def _try_import_instantmesh():
    """Attempt to import InstantMesh from various possible locations."""
    global _HAS_INSTANTMESH, _INSTANTMESH_ERROR

    if not USE_INSTANTMESH:
        _INSTANTMESH_ERROR = "USE_INSTANTMESH env var is not set to 'true'"
        return

    if not INSTANTMESH_CODE_DIR.exists():
        _INSTANTMESH_ERROR = f"InstantMesh code not found at {INSTANTMESH_CODE_DIR}. Run: cd models && git clone https://github.com/TencentARC/InstantMesh.git"
        return

    # Add InstantMesh to Python path
    if str(INSTANTMESH_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(INSTANTMESH_CODE_DIR))

    # Try to find the pipeline or main module
    try:
        # Option 1: They have a pipeline module
        try:
            from instantmesh.pipeline import InstantMeshPipeline
            _HAS_INSTANTMESH = True
            print("[INFO] Imported InstantMeshPipeline from instantmesh.pipeline")
            return
        except ImportError:
            pass

        # Option 2: Direct module import
        try:
            import instantmesh
            _HAS_INSTANTMESH = True
            print("[INFO] Imported instantmesh module")
            return
        except ImportError:
            pass

        # Option 3: Run.py exists (they use scripts)
        run_script = INSTANTMESH_CODE_DIR / "run.py"
        if run_script.exists():
            _HAS_INSTANTMESH = True
            print("[INFO] Found InstantMesh run.py script")
            return

        _INSTANTMESH_ERROR = "InstantMesh code found but could not import any known module. Check the repo structure."

    except Exception as e:
        _INSTANTMESH_ERROR = f"Import error: {str(e)}"
        traceback.print_exc()

_try_import_instantmesh()


class CoarseGenerator:
    """Lazy-loaded singleton for coarse 3D generation."""

    _instance = None
    _pipeline = None
    _loaded = False
    _using_mock = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        """Load model weights into memory."""
        if self._loaded:
            return

        if not USE_INSTANTMESH or not _HAS_INSTANTMESH:
            print(f"[WARN] Running in MOCK mode. Reason: {_INSTANTMESH_ERROR}")
            self._using_mock = True
            self._loaded = True
            return

        print("[INFO] Loading InstantMesh coarse generator...")
        print(f"[INFO] Device: {DEVICE}")
        print(f"[INFO] Weights dir: {INSTANTMESH_WEIGHTS_DIR}")

        try:
            # Try to load the pipeline
            self._pipeline = self._load_pipeline()
            if self._pipeline is not None:
                print("[INFO] Coarse generator loaded successfully.")
                self._loaded = True
            else:
                raise RuntimeError("Pipeline loaded as None")

        except Exception as e:
            print(f"[ERROR] Failed to load InstantMesh: {e}")
            traceback.print_exc()
            print("[WARN] Falling back to MOCK mode.")
            self._using_mock = True
            self._loaded = True

    def _load_pipeline(self):
        """Attempt to load the InstantMesh pipeline."""
        # Try different loading strategies

        # Strategy 1: diffusers-style from_pretrained
        try:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                str(INSTANTMESH_WEIGHTS_DIR),
                torch_dtype=torch.float16 if (FP16 and DEVICE == "cuda") else torch.float32,
            )
            pipe = pipe.to(DEVICE)
            return pipe
        except Exception as e:
            print(f"[INFO] DiffusionPipeline loading failed: {e}")

        # Strategy 2: Try importing from instantmesh module directly
        try:
            import instantmesh
            # Look for common function names
            if hasattr(instantmesh, 'load_pipeline'):
                return instantmesh.load_pipeline(str(INSTANTMESH_WEIGHTS_DIR), device=DEVICE)
            elif hasattr(instantmesh, 'InstantMeshPipeline'):
                pipe = instantmesh.InstantMeshPipeline(str(INSTANTMESH_WEIGHTS_DIR))
                pipe.to(DEVICE)
                return pipe
        except Exception as e:
            print(f"[INFO] instantmesh module loading failed: {e}")

        # Strategy 3: Check if run.py exists and we can use it as a module
        run_script = INSTANTMESH_CODE_DIR / "run.py"
        if run_script.exists():
            print("[INFO] Found run.py - you may need to implement custom inference")

        raise RuntimeError("Could not load InstantMesh pipeline with any known strategy.")

    def generate(
        self,
        image_path: str,
        output_dir: str,
        num_views: int = 6,
        mesh_size: int = 384,
    ) -> Tuple[str, Optional[str]]:
        """
        Generate coarse mesh from image.

        Returns:
            (preview_glb_path, texture_path_or_none)
        """
        self.load()

        if self._using_mock or not _HAS_INSTANTMESH:
            return self._mock_generate(image_path, output_dir)

        return self._real_generate(image_path, output_dir, num_views, mesh_size)

    def _real_generate(
        self,
        image_path: str,
        output_dir: str,
        num_views: int = 6,
        mesh_size: int = 384,
    ) -> Tuple[str, Optional[str]]:
        """Actual InstantMesh inference with MPS fallback."""
        import trimesh

        image = Image.open(image_path).convert("RGB")
        preview_path = os.path.join(output_dir, "preview.glb")

        # Try inference on configured device, fallback to CPU on failure
        device_for_inference = DEVICE
        try:
            mesh = self._run_inference(image, num_views, mesh_size, device_for_inference)
        except RuntimeError as e:
            if "MPS" in str(e) or "Metal" in str(e):
                print(f"[WARN] MPS inference failed ({e}), falling back to CPU...")
                device_for_inference = "cpu"
                mesh = self._run_inference(image, num_views, mesh_size, device_for_inference)
            else:
                raise

        # Export mesh
        if hasattr(mesh, 'export'):
            mesh.export(preview_path)
        else:
            # Convert numpy arrays to mesh if needed
            vertices = mesh.get("vertices", mesh.get("verts"))
            faces = mesh.get("faces", mesh.get("faces"))
            if vertices is not None and faces is not None:
                trimesh.Trimesh(vertices=vertices, faces=faces).export(preview_path)
            else:
                raise ValueError(f"Could not extract mesh data from output type: {type(mesh)}")

        return preview_path, None

    def _run_inference(self, image, num_views, mesh_size, device):
        """Run model inference on specified device."""
        import torch

        with torch.no_grad():
            # Try different pipeline call patterns
            if hasattr(self._pipeline, '__call__'):
                # Standard pipeline call
                try:
                    result = self._pipeline(image, num_views=num_views)
                    if isinstance(result, dict):
                        return result.get("mesh", result)
                    return result
                except TypeError:
                    # Maybe it doesn't take num_views
                    result = self._pipeline(image)
                    if isinstance(result, dict):
                        return result.get("mesh", result)
                    return result

            elif hasattr(self._pipeline, 'generate_mesh'):
                return self._pipeline.generate_mesh(image)

            elif hasattr(self._pipeline, 'run'):
                return self._pipeline.run(image)

            else:
                raise NotImplementedError(
                    f"Pipeline type {type(self._pipeline)} has no known inference method. "
                    "Please check the InstantMesh repo for the correct API."
                )

    def _mock_generate(self, image_path: str, output_dir: str) -> Tuple[str, Optional[str]]:
        """Create a simple icosphere as placeholder."""
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        preview_path = os.path.join(output_dir, "preview.glb")
        mesh.export(preview_path)
        return preview_path, None


# Global singleton
coarse_generator = CoarseGenerator()
