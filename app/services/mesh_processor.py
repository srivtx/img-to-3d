"""Mesh processing utilities for refinement."""

import trimesh
import numpy as np
from pathlib import Path
from typing import Tuple


def subdivide_mesh(mesh_path: str, iterations: int = 1) -> trimesh.Trimesh:
    """
    Subdivide mesh to increase polygon count.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    for _ in range(iterations):
        mesh = mesh.subdivide()
    return mesh


def remesh_smooth(mesh: trimesh.Trimesh, target_faces: int = 50000) -> trimesh.Trimesh:
    """
    Remesh to target face count and smooth.

    The face-count reduction step uses ``simplify_quadric_decimation`` which
    in modern Trimesh delegates to the optional ``fast_simplification``
    package. If that's not installed we fall back to skipping the
    decimation -- the resulting mesh is just higher-poly than requested,
    which is still a perfectly usable GLB. Smoothing and UV generation are
    pure-trimesh and always run.
    """
    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(target_faces)
        except (ImportError, ModuleNotFoundError, ValueError) as e:
            print(
                f"[refinement] simplify_quadric_decimation skipped "
                f"({type(e).__name__}: {e}); keeping {len(mesh.faces)} faces"
            )

    # Taubin smoothing (preserves volume better than Laplacian)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
    return mesh


def enhance_uvs(mesh: trimesh.Trimesh, resolution: int = 1024) -> trimesh.Trimesh:
    """
    Ensure mesh has proper UV coordinates for texturing.
    If missing, generate basic spherical UVs.
    """
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
        return mesh
    
    # Generate spherical UV mapping
    vertices = mesh.vertices - mesh.centroid
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normalized = vertices / norms
    
    u = 0.5 + np.arctan2(normalized[:, 0], normalized[:, 2]) / (2 * np.pi)
    v = 0.5 - np.arcsin(np.clip(normalized[:, 1], -1.0, 1.0)) / np.pi
    
    mesh.visual.uv = np.stack([u, v], axis=-1)
    return mesh


def process_mesh(
    input_path: str,
    output_path: str,
    subdivisions: int = 1,
    target_faces: int = 50000,
    texture_resolution: int = 1024,
) -> str:
    """
    Full refinement pipeline:
    1. Subdivide
    2. Remesh/smooth
    3. Fix UVs
    4. Export GLB
    """
    mesh = subdivide_mesh(input_path, iterations=subdivisions)
    mesh = remesh_smooth(mesh, target_faces=target_faces)
    mesh = enhance_uvs(mesh, resolution=texture_resolution)
    
    # Export as GLB
    mesh.export(output_path)
    return output_path
