"""Refinement pipeline: takes coarse mesh and improves quality."""

import os
from pathlib import Path
from typing import Optional

from app.services.mesh_processor import process_mesh
from app.core.config import REFINE_SUBDIVISIONS, TEXTURE_RESOLUTION


class RefinementPipeline:
    """Async refinement of coarse mesh."""
    
    def refine(
        self,
        coarse_path: str,
        output_dir: str,
    ) -> str:
        """
        Refine a coarse mesh into higher quality output.
        
        Returns path to final GLB.
        """
        final_path = os.path.join(output_dir, "final.glb")
        
        process_mesh(
            input_path=coarse_path,
            output_path=final_path,
            subdivisions=REFINE_SUBDIVISIONS,
            target_faces=50000,
            texture_resolution=TEXTURE_RESOLUTION,
        )
        
        return final_path


refinement_pipeline = RefinementPipeline()
