"""Mesh processing utilities for refinement.

The refinement pipeline takes the coarse ``preview.glb`` produced by
InstantMesh and improves it slightly before exporting ``final.glb``. The
key constraint is preserving the per-vertex colors that InstantMesh bakes
into the preview -- those colors *are* the texture for our render. Naive
mesh operations (Trimesh's subdivide, replacing the visual with UVs) drop
that data and produce a flat-white mesh, which is what was happening
before this refactor.

Two refinement paths:

* **Color path** (preview has vertex colors): Taubin smoothing only, plus
  optional color-aware simplification. Keeps the GLB visually identical
  to the preview but cleaner-looking.
* **Geometric path** (no color data, e.g. a mesh from a different
  pipeline): subdivide -> simplify -> smooth -> generate spherical UVs.

We pick the path automatically based on whether ``mesh.visual.kind`` is
``"vertex"`` / ``"face"`` (color data) or ``"texture"`` / ``None``.
"""

import trimesh
import numpy as np
from typing import Optional


def _has_color_visuals(mesh: trimesh.Trimesh) -> bool:
    """True if the mesh's visual carries actual per-vertex/face colors."""
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return False
    kind = getattr(visual, "kind", None)
    return kind in ("vertex", "face")


def _try_simplify(
    mesh: trimesh.Trimesh, target_faces: int
) -> trimesh.Trimesh:
    """Best-effort face reduction. Returns mesh unchanged on any failure.

    Trimesh ``simplify_quadric_decimation`` takes ``face_count=`` (or
    ``percent=``) keyword args, NOT a positional ``target_faces``. Calling
    it positionally treats the int as ``percent`` and raises
    "target_reduction must be between 0 and 1". We pass face_count
    correctly and fall through gracefully on missing fast_simplification
    or any other error -- a higher-poly mesh is still a valid GLB.
    """
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except (ImportError, ModuleNotFoundError, ValueError, TypeError) as e:
        print(
            f"[refinement] simplify_quadric_decimation skipped "
            f"({type(e).__name__}: {e}); keeping {len(mesh.faces)} faces"
        )
        return mesh


def subdivide_mesh(
    mesh: trimesh.Trimesh, iterations: int = 1
) -> trimesh.Trimesh:
    """Subdivide each triangle into 4. Used in the geometric (no-color) path
    only -- Trimesh's subdivide does not interpolate vertex colors, so we
    avoid it for color-bearing meshes.
    """
    for _ in range(iterations):
        mesh = mesh.subdivide()
    return mesh


def smooth_inplace(mesh: trimesh.Trimesh, iterations: int = 10) -> None:
    """Taubin smoothing -- preserves volume better than Laplacian and (key
    for our case) preserves vertex colors and the visual type."""
    trimesh.smoothing.filter_taubin(
        mesh, lamb=0.5, nu=-0.53, iterations=iterations
    )


def enhance_uvs(
    mesh: trimesh.Trimesh, resolution: int = 1024
) -> trimesh.Trimesh:
    """Generate basic spherical UVs **only if** the mesh has neither UVs nor
    color data. Setting ``mesh.visual.uv`` converts the visual to
    ``TextureVisuals``, which has no concept of vertex_colors -- so for a
    color-bearing mesh we'd be silently dropping the chair/bag/etc.'s
    appearance. Skip in that case.
    """
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return mesh

    uv = getattr(visual, "uv", None)
    if uv is not None and len(uv) > 0:
        return mesh

    if _has_color_visuals(mesh):
        return mesh

    # Plain mesh with no visual data -> assign spherical UVs as a starting
    # point for downstream texturing.
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
    """Refine ``input_path`` and write ``output_path``.

    Picks the right path based on whether the input has color data:

      * Has colors  (InstantMesh preview): smooth, optionally simplify.
      * No colors   (geometric mesh):      subdivide, simplify, smooth, UVs.
    """
    mesh = trimesh.load(input_path, force="mesh")

    if _has_color_visuals(mesh):
        # Color-preserving path. Skip subdivide (would drop colors) and
        # skip UV generation (would replace ColorVisuals with TextureVisuals
        # and drop colors). Smooth and try to simplify.
        smooth_inplace(mesh, iterations=10)
        # Simplification *may* drop colors depending on the trimesh /
        # fast_simplification version. We try it but fall back to the
        # un-simplified mesh if colors are lost so final.glb always has
        # the chair/bag/etc.'s appearance -- a slightly higher-poly file
        # is a better outcome than a flat-white one.
        original_color_kind = mesh.visual.kind
        candidate = _try_simplify(mesh, target_faces=target_faces)
        if candidate is not mesh and getattr(
            candidate.visual, "kind", None
        ) == original_color_kind:
            mesh = candidate
        elif candidate is not mesh:
            print(
                "[refinement] simplification dropped color data; keeping "
                "smoothed-but-unsimplified mesh to preserve appearance"
            )
    else:
        # Geometric path. Original behavior: subdivide -> simplify -> smooth
        # -> spherical UVs. Useful for non-InstantMesh inputs that come in
        # without any visual data.
        mesh = subdivide_mesh(mesh, iterations=subdivisions)
        mesh = _try_simplify(mesh, target_faces=target_faces)
        smooth_inplace(mesh, iterations=10)
        mesh = enhance_uvs(mesh, resolution=texture_resolution)

    mesh.export(output_path)
    return output_path
