"""InstantMesh in-process coarse generator.

This module wraps TencentARC/InstantMesh as a singleton that loads the
diffusion + reconstruction models once into the current Python process and
reuses them for every request. If anything is missing (env flag off, repo
not cloned, weights not downloaded, optional CUDA-only deps unavailable)
it transparently falls back to a mock icosphere so the rest of the app
keeps working.

The integration follows the official `InstantMesh/app.py` reference
(https://github.com/TencentARC/InstantMesh/blob/main/app.py) which is the
canonical end-to-end pattern (image -> 6 zero123plus views -> triplane
-> mesh -> GLB). We override the config's checkpoint paths to point at
our own ``models/instantmesh/`` weights dir so we never trigger an
unattended ``hf_hub_download`` at inference time.

Environment variables:
    USE_INSTANTMESH:               "true" to enable real inference
    DEVICE:                        "cuda" / "mps" / "cpu" (real path is CUDA-only)
    INSTANTMESH_CONFIG:            yaml stem in InstantMesh/configs/, default "instant-mesh-large"
    INSTANTMESH_DIFFUSION_STEPS:   denoise steps (default 75)
    INSTANTMESH_REMBG:             "true"/"false" - run rembg pre-processing (default "true")
    INSTANTMESH_SEED:              int seed for reproducibility (default 42)
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from app.core.config import DEVICE, FP16


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
USE_INSTANTMESH = os.getenv("USE_INSTANTMESH", "false").lower() == "true"
DIFFUSION_STEPS = int(os.getenv("INSTANTMESH_DIFFUSION_STEPS", "75"))
DEFAULT_CONFIG = os.getenv("INSTANTMESH_CONFIG", "instant-mesh-large")
DO_REMBG = os.getenv("INSTANTMESH_REMBG", "true").lower() == "true"
SEED = int(os.getenv("INSTANTMESH_SEED", "42"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INSTANTMESH_CODE_DIR = PROJECT_ROOT / "models" / "InstantMesh"
INSTANTMESH_WEIGHTS_DIR = PROJECT_ROOT / "models" / "instantmesh"

# Mapping from config name -> reconstruction checkpoint filename in the HF repo
WEIGHT_FILES = {
    "instant-mesh-base": "instant_mesh_base.ckpt",
    "instant-mesh-large": "instant_mesh_large.ckpt",
    "instant-nerf-base": "instant_nerf_base.ckpt",
    "instant-nerf-large": "instant_nerf_large.ckpt",
}
UNET_FILE = "diffusion_pytorch_model.bin"


def _shim_transformers_compat() -> None:
    """Re-add helpers that newer transformers (5.x) removed.

    InstantMesh's vendored DiNo encoder
    (``src/models/encoder/dino.py``) does:

        from transformers.pytorch_utils import (
            find_pruneable_heads_and_indices,
            prune_linear_layer,
        )

    and at runtime calls ``self.get_head_mask(...)`` via the
    ``PreTrainedModel`` base class. All three were public-but-internal
    helpers in transformers <=4.x and were removed/relocated in 5.x as
    part of an internal cleanup. Modern Colab installs transformers
    5.x by default, so:

      * the ``pytorch_utils`` imports break at module-load time
        (caught at ``instantiate_from_config(config.model_config)``)
      * the ``get_head_mask`` call breaks at inference time inside
        ``ViTModel.forward`` (caught at ``model.forward_planes``)

    Reimplementing them locally and injecting them into
    ``transformers.pytorch_utils`` / ``PreTrainedModel`` avoids forcing
    a transformers downgrade (which is slow and risks knock-on
    dependency conflicts with diffusers / accelerate / huggingface-hub).
    The implementations here are lifted verbatim from transformers
    v4.46.0 -- they're tiny, pure-PyTorch helpers with no
    transformers-version-specific behavior.
    """
    try:
        import transformers.pytorch_utils as tpu  # type: ignore
        from transformers import PreTrainedModel  # type: ignore
    except Exception:
        # transformers itself isn't installed yet; nothing to shim. Whoever
        # called us is about to fail at the next import anyway.
        return

    from torch import nn

    if not hasattr(tpu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(  # type: ignore[misc]
            heads, n_heads, head_size, already_pruned_heads
        ):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        tpu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(tpu, "prune_linear_layer"):
        def prune_linear_layer(layer: "nn.Linear", index, dim: int = 0) -> "nn.Linear":  # type: ignore[misc]
            index = index.to(layer.weight.device)
            W = layer.weight.index_select(dim, index).clone().detach()
            if layer.bias is not None:
                if dim == 1:
                    b = layer.bias.clone().detach()
                else:
                    b = layer.bias[index].clone().detach()
            new_size = list(layer.weight.size())
            new_size[dim] = len(index)
            new_layer = nn.Linear(
                new_size[1], new_size[0], bias=layer.bias is not None
            ).to(layer.weight.device)
            new_layer.weight.requires_grad = False
            new_layer.weight.copy_(W.contiguous())
            new_layer.weight.requires_grad = True
            if layer.bias is not None:
                new_layer.bias.requires_grad = False
                new_layer.bias.copy_(b.contiguous())
                new_layer.bias.requires_grad = True
            return new_layer

        tpu.prune_linear_layer = prune_linear_layer

    if not hasattr(PreTrainedModel, "get_head_mask"):
        def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=self.dtype)
            return head_mask

        def get_head_mask(
            self, head_mask, num_hidden_layers, is_attention_chunked=False
        ):
            if head_mask is not None:
                head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                if is_attention_chunked is True:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d  # type: ignore[attr-defined]
        PreTrainedModel.get_head_mask = get_head_mask  # type: ignore[attr-defined]


def _check_setup() -> Tuple[bool, str]:
    """Cheap pre-flight check. Returns (ok, message)."""
    if not USE_INSTANTMESH:
        return False, "USE_INSTANTMESH env var is not 'true'"
    if DEFAULT_CONFIG not in WEIGHT_FILES:
        return False, f"Unknown INSTANTMESH_CONFIG '{DEFAULT_CONFIG}' (valid: {list(WEIGHT_FILES)})"
    if not INSTANTMESH_CODE_DIR.exists():
        return False, f"InstantMesh repo not cloned at {INSTANTMESH_CODE_DIR}"
    if not (INSTANTMESH_CODE_DIR / "src").exists():
        return False, f"InstantMesh src/ missing in {INSTANTMESH_CODE_DIR}"
    cfg_path = INSTANTMESH_CODE_DIR / "configs" / f"{DEFAULT_CONFIG}.yaml"
    if not cfg_path.exists():
        return False, f"Config file missing: {cfg_path}"
    if not INSTANTMESH_WEIGHTS_DIR.exists():
        return False, f"Weights dir missing: {INSTANTMESH_WEIGHTS_DIR}"

    needed = [UNET_FILE, WEIGHT_FILES[DEFAULT_CONFIG]]
    missing = [n for n in needed if not (INSTANTMESH_WEIGHTS_DIR / n).exists()]
    if missing:
        return False, f"Missing weight files in {INSTANTMESH_WEIGHTS_DIR}: {missing}"
    return True, f"OK (config={DEFAULT_CONFIG}, weights at {INSTANTMESH_WEIGHTS_DIR})"


class CoarseGenerator:
    """Lazy, thread-safe singleton wrapping the InstantMesh pipeline."""

    _instance: Optional["CoarseGenerator"] = None
    _load_lock = threading.Lock()
    _gpu_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
            cls._instance._using_mock = False
            cls._instance._setup_ok = False
            cls._instance._setup_msg = ""
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load model into VRAM; idempotent and thread-safe."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return

            self._setup_ok, self._setup_msg = _check_setup()
            if not self._setup_ok:
                print(f"[InstantMesh] MOCK mode: {self._setup_msg}")
                self._using_mock = True
                self._loaded = True
                return

            print(f"[InstantMesh] Setup: {self._setup_msg}")
            print(f"[InstantMesh] Device={DEVICE} FP16={FP16} steps={DIFFUSION_STEPS} rembg={DO_REMBG}")

            try:
                self._load_real_model()
                print("[InstantMesh] REAL model loaded successfully")
            except Exception as e:  # broad: any import / IO / CUDA error -> mock
                print(f"[InstantMesh] Real load FAILED -> falling back to MOCK: {e}")
                traceback.print_exc()
                self._using_mock = True

            self._loaded = True

    def generate(
        self,
        image_path: str,
        output_dir: str,
        num_views: int = 6,
        mesh_size: int = 384,  # noqa: ARG002 - kept for API stability
    ) -> Tuple[str, Optional[str]]:
        """Run end-to-end image-to-3D and return (preview_glb_path, texture_path|None)."""
        self.load()
        if self._using_mock:
            return self._mock_generate(image_path, output_dir)

        try:
            with self._gpu_lock:
                return self._real_generate(image_path, output_dir, num_views)
        except Exception as e:
            print(f"[InstantMesh] Real inference FAILED -> falling back to MOCK for this job: {e}")
            traceback.print_exc()
            return self._mock_generate(image_path, output_dir)

    # ------------------------------------------------------------------
    # Internals: model loading
    # ------------------------------------------------------------------
    def _load_real_model(self) -> None:
        # Make `import src.utils...` (the way InstantMesh code is laid out) work.
        if str(INSTANTMESH_CODE_DIR) not in sys.path:
            sys.path.insert(0, str(INSTANTMESH_CODE_DIR))

        # Backfill helpers that newer transformers removed before any
        # InstantMesh code path imports them. Must run BEFORE we touch
        # `src.models.lrm_mesh` (loaded indirectly via instantiate_from_config
        # below) which transitively imports
        # `transformers.pytorch_utils.find_pruneable_heads_and_indices`,
        # and also patches `PreTrainedModel.get_head_mask` so that the
        # vendored ViTModel forward pass works at inference time.
        _shim_transformers_compat()

        # All third-party imports live INSIDE this method so that a missing
        # optional dep (e.g. nvdiffrast on a non-CUDA box) raises here and is
        # caught by the outer try/except in load(), not at module import time.
        from omegaconf import OmegaConf
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        import rembg

        from src.utils.train_util import instantiate_from_config  # type: ignore

        cfg_path = INSTANTMESH_CODE_DIR / "configs" / f"{DEFAULT_CONFIG}.yaml"
        config = OmegaConf.load(str(cfg_path))

        # Point the config at OUR weights dir so the `os.path.exists(...)`
        # checks in InstantMesh code resolve locally instead of triggering
        # an unattended HF download.
        config.infer_config.unet_path = str(INSTANTMESH_WEIGHTS_DIR / UNET_FILE)
        config.infer_config.model_path = str(
            INSTANTMESH_WEIGHTS_DIR / WEIGHT_FILES[DEFAULT_CONFIG]
        )

        self.config = config
        self.config_name = DEFAULT_CONFIG
        self.is_flexicubes = DEFAULT_CONFIG.startswith("instant-mesh")

        device = torch.device(DEVICE)
        torch_dtype = torch.float16 if (FP16 and device.type == "cuda") else torch.float32

        # ---- Diffusion pipeline (zero123plus + custom UNet) ----------
        # InstantMesh's reference run.py passes custom_pipeline="zero123plus".
        # That short name makes Diffusers look in the community-pipelines
        # mirror at v{diffusers_version}/zero123plus.py -- which 404s on
        # modern Diffusers because zero123plus.py was never published to
        # that mirror at recent versions. Fortunately InstantMesh vendors
        # its own copy of pipeline.py at zero123plus/pipeline.py inside
        # the cloned repo, so we point custom_pipeline at that absolute
        # local file. Diffusers will load the class from disk and skip
        # the network lookup entirely.
        local_pipeline_py = INSTANTMESH_CODE_DIR / "zero123plus" / "pipeline.py"
        if local_pipeline_py.is_file():
            custom_pipeline_arg: str = str(local_pipeline_py)
            print(
                f"[InstantMesh] Using vendored zero123plus pipeline at "
                f"{custom_pipeline_arg}"
            )
        else:
            # Fall back to the upstream short name -- will fail on modern
            # Diffusers but at least the error message is informative.
            custom_pipeline_arg = "zero123plus"
            print(
                f"[InstantMesh] WARNING: vendored pipeline.py not found at "
                f"{local_pipeline_py}, falling back to community pipeline name"
            )

        print(f"[InstantMesh] Loading zero123plus pipeline (dtype={torch_dtype})")
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline=custom_pipeline_arg,
            torch_dtype=torch_dtype,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

        print(f"[InstantMesh] Loading custom UNet from {config.infer_config.unet_path}")
        unet_state = torch.load(config.infer_config.unet_path, map_location="cpu")
        pipeline.unet.load_state_dict(unet_state, strict=True)
        del unet_state
        pipeline = pipeline.to(device)

        # ---- Reconstruction model -----------------------------------
        print(f"[InstantMesh] Loading recon model from {config.infer_config.model_path}")
        model = instantiate_from_config(config.model_config)
        ckpt = torch.load(config.infer_config.model_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # Strip the `lrm_generator.` prefix used during training and drop
        # `source_camera` keys that the inference model doesn't expose.
        prefix = "lrm_generator."
        state = {
            k[len(prefix):]: v
            for k, v in state.items()
            if k.startswith(prefix) and "source_camera" not in k
        }
        model.load_state_dict(state, strict=True)
        del ckpt, state

        model = model.to(device)
        if self.is_flexicubes:
            model.init_flexicubes_geometry(device, fovy=30.0)
        model = model.eval()

        self.pipeline = pipeline
        self.model = model
        self.device = device
        self.torch_dtype = torch_dtype
        self.rembg_session = rembg.new_session() if DO_REMBG else None

    # ------------------------------------------------------------------
    # Internals: inference
    # ------------------------------------------------------------------
    def _real_generate(
        self,
        image_path: str,
        output_dir: str,
        num_views: int,
    ) -> Tuple[str, Optional[str]]:
        from torchvision.transforms import v2
        from einops import rearrange
        from pytorch_lightning import seed_everything

        from src.utils.camera_util import get_zero123plus_input_cameras  # type: ignore
        from src.utils.mesh_util import save_glb  # type: ignore
        from src.utils.infer_util import remove_background, resize_foreground  # type: ignore

        os.makedirs(output_dir, exist_ok=True)
        preview_path = os.path.join(output_dir, "preview.glb")

        seed_everything(SEED)

        # ----- Stage 1: input image -> 6-view tile (zero123plus) --------
        input_image = Image.open(image_path)
        if DO_REMBG:
            input_image = remove_background(input_image, self.rembg_session)
            input_image = resize_foreground(input_image, 0.85)
        elif input_image.mode != "RGBA":
            # resize_foreground requires RGBA, but if rembg is off and the
            # image already has no alpha, fabricate a fully-opaque alpha.
            input_image = input_image.convert("RGBA")

        # The pipeline may have been offloaded to CPU at the end of the
        # previous request (see VRAM cleanup below). Move it back before
        # running diffusion -- a ~1-2 second cost on T4. No-op if it's
        # already on the target device.
        if self.device.type == "cuda":
            try:
                self.pipeline.to(self.device)
            except Exception:
                pass

        with torch.no_grad():
            mv_image = self.pipeline(
                input_image,
                num_inference_steps=DIFFUSION_STEPS,
            ).images[0]

        # Best-effort dump for debugging; never break inference if this fails.
        try:
            mv_image.save(os.path.join(output_dir, "multiview.png"))
            input_image.save(os.path.join(output_dir, "input_processed.png"))
        except Exception:
            pass

        # We're done with the diffusion pipeline for this request -- offload
        # it to CPU and reclaim the ~3 GB of VRAM it's hogging. The
        # reconstruction model below (FlexiCubes mesh extraction in
        # particular) is memory-hungry and needs every byte we can spare on
        # smaller GPUs (Colab T4 = 16 GB). The next request will move the
        # pipeline back to CUDA before generating views -- a ~1-2 second
        # round-trip that's negligible compared to the 50 s diffusion step.
        if self.device.type == "cuda":
            try:
                self.pipeline.to("cpu")
            except Exception:
                pass
            torch.cuda.empty_cache()

        # ----- Stage 2: 6 views -> triplane -> mesh ---------------------
        images_np = np.asarray(mv_image, dtype=np.float32) / 255.0
        images_t = torch.from_numpy(images_np).permute(2, 0, 1).contiguous().float()
        images_t = rearrange(images_t, "c (n h) (m w) -> (n m) c h w", n=3, m=2)
        # images_t now has shape (6, 3, 320, 320) approximately

        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(self.device)

        images_t = images_t.unsqueeze(0).to(self.device)
        images_t = v2.functional.resize(
            images_t, (320, 320), interpolation=3, antialias=True
        ).clamp(0, 1)

        if num_views == 4:
            view_idx = torch.tensor([0, 2, 4, 5], device=self.device).long()
            images_t = images_t[:, view_idx]
            input_cameras = input_cameras[:, view_idx]

        with torch.no_grad():
            planes = self.model.forward_planes(images_t, input_cameras)
            mesh_out = self.model.extract_mesh(
                planes,
                use_texture_map=False,
                **self.config.infer_config,
            )

        vertices, faces, vertex_colors = mesh_out
        # InstantMesh's `app.py` reorients axes to glTF/Three.js (Y-up) before
        # passing to `save_glb` (which itself flips X/Z to make a right-handed
        # glTF). We follow the same recipe.
        vertices = vertices[:, [1, 2, 0]]
        save_glb(vertices, faces, vertex_colors, preview_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[InstantMesh] Wrote {preview_path}")
        return preview_path, None

    # ------------------------------------------------------------------
    # Mock fallback
    # ------------------------------------------------------------------
    def _mock_generate(self, image_path: str, output_dir: str) -> Tuple[str, Optional[str]]:
        import trimesh

        os.makedirs(output_dir, exist_ok=True)
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        preview_path = os.path.join(output_dir, "preview.glb")
        mesh.export(preview_path)
        return preview_path, None


# Module-level singleton (preserves the existing public import surface).
coarse_generator = CoarseGenerator()
