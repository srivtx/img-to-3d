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
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Tuple, Optional

import torch
from PIL import Image

from app.core.config import DEVICE, FP16

# ============================================================================
# CONFIGURATION
# ============================================================================
USE_INSTANTMESH = os.getenv("USE_INSTANTMESH", "false").lower() == "true"

# Path to InstantMesh code (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INSTANTMESH_CODE_DIR = PROJECT_ROOT / "models" / "InstantMesh"
INSTANTMESH_WEIGHTS_DIR = PROJECT_ROOT / "models" / "instantmesh"


def _check_instantmesh_setup() -> Tuple[bool, str]:
    """Check if InstantMesh is properly set up."""
    if not USE_INSTANTMESH:
        return False, "USE_INSTANTMESH env var is not set to 'true'"

    if not INSTANTMESH_CODE_DIR.exists():
        return False, f"InstantMesh code not found at {INSTANTMESH_CODE_DIR}"

    if not INSTANTMESH_WEIGHTS_DIR.exists():
        return False, f"Weights not found at {INSTANTMESH_WEIGHTS_DIR}"

    weight_files = list(INSTANTMESH_WEIGHTS_DIR.iterdir())
    if len(weight_files) == 0:
        return False, f"Weights directory is empty: {INSTANTMESH_WEIGHTS_DIR}"

    return True, f"Found {len(weight_files)} weight files"


class CoarseGenerator:
    """Lazy-loaded singleton for coarse 3D generation."""

    _instance = None
    _loaded = False
    _using_mock = False
    _setup_ok = False
    _setup_msg = ""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        """Load model weights into memory."""
        if self._loaded:
            return

        self._setup_ok, self._setup_msg = _check_instantmesh_setup()

        if not self._setup_ok:
            print(f"[WARN] Running in MOCK mode. Reason: {self._setup_msg}")
            self._using_mock = True
            self._loaded = True
            return

        print("[INFO] InstantMesh setup OK:", self._setup_msg)
        print("[INFO] Loading InstantMesh coarse generator...")
        print(f"[INFO] Device: {DEVICE}")
        print(f"[INFO] Weights dir: {INSTANTMESH_WEIGHTS_DIR}")

        # Check what files are in the weights dir
        try:
            weight_files = list(INSTANTMESH_WEIGHTS_DIR.iterdir())
            print(f"[INFO] Weight files: {[f.name for f in weight_files[:5]]}")
        except Exception as e:
            print(f"[WARN] Could not list weights: {e}")

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

        if self._using_mock or not self._setup_ok:
            return self._mock_generate(image_path, output_dir)

        return self._real_generate(image_path, output_dir, num_views, mesh_size)

    def _real_generate(
        self,
        image_path: str,
        output_dir: str,
        num_views: int = 6,
        mesh_size: int = 384,
    ) -> Tuple[str, Optional[str]]:
        """Run InstantMesh via subprocess call."""
        preview_path = os.path.join(output_dir, "preview.glb")

        # Strategy: Run InstantMesh's run.py or gradio_app.py via subprocess
        # This is the most reliable way since their API may change

        run_script = INSTANTMESH_CODE_DIR / "run.py"
        gradio_script = INSTANTMESH_CODE_DIR / "gradio_app.py"

        if run_script.exists():
            return self._run_via_script(run_script, image_path, preview_path)
        elif gradio_script.exists():
            return self._run_via_script(gradio_script, image_path, preview_path)
        else:
            print("[WARN] No run.py or gradio_app.py found. Falling back to mock.")
            return self._mock_generate(image_path, output_dir)

    def _run_via_script(self, script_path: Path, image_path: str, output_path: str) -> Tuple[str, Optional[str]]:
        """Run InstantMesh script to generate mesh."""
        import shutil

        # Create a temp directory for the script output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to run the script
            # Common patterns:
            # python run.py --input_image path --output_dir path
            # python gradio_app.py (need to modify or call functions)

            # First, let's try a direct approach by importing and running
            try:
                return self._run_via_import(image_path, output_path)
            except Exception as e:
                print(f"[INFO] Import approach failed: {e}")

            # Fallback: Try subprocess with common args
            cmd = [
                sys.executable,
                str(script_path),
                "--input", image_path,
                "--output", tmpdir,
                "--device", DEVICE,
            ]

            print(f"[INFO] Running: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(INSTANTMESH_CODE_DIR),
                )

                if result.returncode == 0:
                    # Find the generated mesh
                    mesh_files = list(Path(tmpdir).glob("*.obj")) + list(Path(tmpdir).glob("*.glb")) + list(Path(tmpdir).glob("*.ply"))
                    if mesh_files:
                        import trimesh
                        mesh = trimesh.load(str(mesh_files[0]))
                        mesh.export(output_path)
                        print(f"[INFO] Generated mesh: {output_path}")
                        return output_path, None
                else:
                    print(f"[WARN] Script failed with code {result.returncode}")
                    print(f"[WARN] stderr: {result.stderr[:500]}")

            except subprocess.TimeoutExpired:
                print("[WARN] Script timed out after 120 seconds")
            except Exception as e:
                print(f"[WARN] Script execution failed: {e}")

        # If all else fails, mock
        print("[WARN] All generation methods failed. Using mock.")
        return self._mock_generate(image_path, os.path.dirname(output_path))

    def _run_via_import(self, image_path: str, output_path: str) -> Tuple[str, Optional[str]]:
        """Try to import and run InstantMesh directly."""
        # Save current sys.path
        original_path = sys.path.copy()

        try:
            # Add InstantMesh to path
            if str(INSTANTMESH_CODE_DIR) not in sys.path:
                sys.path.insert(0, str(INSTANTMESH_CODE_DIR))

            # Try to find and import their inference function
            # This is highly dependent on their code structure

            # Look for common patterns
            if (INSTANTMESH_CODE_DIR / "instantmesh" / "pipeline.py").exists():
                from instantmesh.pipeline import InstantMeshPipeline
                pipe = InstantMeshPipeline.from_pretrained(str(INSTANTMESH_WEIGHTS_DIR))
                pipe.to(DEVICE)

                image = Image.open(image_path).convert("RGB")
                result = pipe(image)

                if isinstance(result, dict) and "mesh" in result:
                    mesh = result["mesh"]
                else:
                    mesh = result

                mesh.export(output_path)
                return output_path, None

        except Exception as e:
            print(f"[INFO] Import approach error: {e}")
            raise

        finally:
            # Restore sys.path
            sys.path = original_path

    def _mock_generate(self, image_path: str, output_dir: str) -> Tuple[str, Optional[str]]:
        """Create a simple icosphere as placeholder."""
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        preview_path = os.path.join(output_dir, "preview.glb")
        mesh.export(preview_path)
        return preview_path, None


# Global singleton
coarse_generator = CoarseGenerator()
