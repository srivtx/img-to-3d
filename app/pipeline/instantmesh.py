"""
InstantMesh coarse generation wrapper.

SETUP REQUIRED FOR REAL INFERENCE:
    cd models
    git clone https://github.com/TencentARC/InstantMesh.git
    cd InstantMesh
    pip install -r requirements.txt
    # Download weights from their HuggingFace and place in models/instantmesh/

Then set USE_INSTANTMESH = True below.
"""

import os
import sys
import warnings
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

if USE_INSTANTMESH and INSTANTMESH_CODE_DIR.exists():
    sys.path.insert(0, str(INSTANTMESH_CODE_DIR))
    try:
        from src.models.mesh_fusion import quick_voxel_and_mesh
        from src.models.lgm_model import LGMModel
        from src.utils import seed_everything, get_device
        _HAS_INSTANTMESH = True
    except ImportError as e:
        _INSTANTMESH_ERROR = str(e)
        warnings.warn(f"InstantMesh code found but import failed: {e}")
else:
    _INSTANTMESH_ERROR = (
        "InstantMesh not set up. "
        f"Clone to {INSTANTMESH_CODE_DIR} and set USE_INSTANTMESH=True. "
        f"Current code dir exists: {INSTANTMESH_CODE_DIR.exists()}"
    )


class CoarseGenerator:
    """Lazy-loaded singleton for coarse 3D generation."""

    _instance = None
    _model = None
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
            # NOTE: This is a template — adapt to InstantMesh's actual API
            # after cloning and inspecting their code.
            # Common patterns:
            #   - Load a config YAML
            #   - Instantiate a pipeline class
            #   - Move to device
            #   - Set eval mode

            # Example (will need adjustment based on their actual API):
            # self._model = LGMModel.from_pretrained(str(INSTANTMESH_WEIGHTS_DIR))
            # self._model.to(DEVICE)
            # if FP16 and DEVICE == "cuda":
            #     self._model.half()
            # self._model.eval()

            print("[INFO] Coarse generator loaded.")
            self._loaded = True

        except Exception as e:
            print(f"[ERROR] Failed to load InstantMesh: {e}")
            print("[WARN] Falling back to MOCK mode.")
            self._using_mock = True
            self._loaded = True

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

        # Try inference on configured device, fallback to CPU on failure
        device_for_inference = DEVICE
        try:
            outputs = self._run_inference(image, num_views, mesh_size, device_for_inference)
        except RuntimeError as e:
            if "MPS" in str(e) or "Metal" in str(e):
                print(f"[WARN] MPS inference failed ({e}), falling back to CPU...")
                device_for_inference = "cpu"
                outputs = self._run_inference(image, num_views, mesh_size, device_for_inference)
            else:
                raise

        mesh = outputs["mesh"]
        preview_path = os.path.join(output_dir, "preview.glb")

        # Ensure it's a trimesh object
        if hasattr(mesh, 'export'):
            mesh.export(preview_path)
        else:
            # Convert if needed
            trimesh.Trimesh(vertices=outputs["vertices"], faces=outputs["faces"]).export(preview_path)

        return preview_path, None

    def _run_inference(self, image, num_views, mesh_size, device):
        """Run model inference on specified device."""
        # TODO: Replace with actual InstantMesh forward pass
        # This is a placeholder showing the pattern:
        #
        # with torch.no_grad():
        #     if hasattr(self._model, 'generate_mesh'):
        #         return self._model.generate_mesh(
        #             image,
        #             num_views=num_views,
        #             mesh_size=mesh_size,
        #             device=device
        #         )
        #     else:
        #         inputs = self._preprocess(image).to(device)
        #         outputs = self._model(inputs)
        #         return self._postprocess(outputs)
        raise NotImplementedError(
            "Real inference not yet wired. "
            "Please inspect the cloned InstantMesh repo and implement _run_inference."
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
