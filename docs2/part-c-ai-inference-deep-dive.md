# Part C: How AI Inference Actually Works — Line by Line

## Before We Start

This is the deep dive. We're going to trace through what happens when you upload a photo and get back a 3D model. Not at a high level. **Line by line.**

**Prerequisites:** You've read Part A and B. You know what vertices are, what async is, and how polling works.

---

## Chapter 1: The Big Picture (Then We Zoom In)

```
Your Photo (512x512 RGB)
    ↓
[InstantMesh Pipeline]
    ├─ Step 1: Background Removal (rembg/U2Net)
    ├─ Step 2: Multi-View Generation (Zero123++ diffusion)
    ├─ Step 3: 3D Reconstruction (LRM — Large Reconstruction Model)
    ├─ Step 4: Mesh Extraction (FlexiCubes)
    └─ Step 5: Export to GLB
    ↓
3D Model File (GLB)
```

Each step is a neural network or algorithm. Together they form a **pipeline**.

**Total time on T4 GPU:** ~30-60 seconds  
**Total parameters loaded:** ~2-3 billion  
**VRAM peak:** ~12 GB

---

## Chapter 2: Step 1 — Background Removal

### Why Remove Background?

If you photograph a shoe on a carpet, the AI might try to reconstruct the carpet too. We want **only the object**.

### What is rembg?

`rembg` is a Python package that uses a neural network called **U2Net** to segment (cut out) the foreground object.

### Exercise 2.1: Try rembg Yourself

```python
from PIL import Image
import rembg

# Load a photo
input_image = Image.open("your_photo.jpg")

# Create a session (loads the model)
session = rembg.new_session()

# Remove background
output = rembg.remove(input_image, session=session)

# Save
output.save("no_background.png")
```

**What just happened:**
1. Loaded a U2Net model (~170 MB)
2. The model looked at your photo
3. For each pixel, it predicted: "foreground" or "background"
4. Background pixels became transparent (alpha = 0)
5. Foreground pixels kept their color

### How U2Net Works (Simplified)

U2Net is a **U-shaped convolutional network**:

```
Input Image (3 channels: RGB)
    ↓
Encoder (downsampling)
    ├─ Block 1: 512×512 → 256×256 (detect edges)
    ├─ Block 2: 256×256 → 128×128 (detect textures)
    ├─ Block 3: 128×128 → 64×64   (detect parts)
    ├─ Block 4: 64×64 → 32×32     (detect objects)
    └─ Block 5: 32×32 → 16×16     (global context)
    ↓
Decoder (upsampling)
    ├─ Block 5: 16×16 → 32×32
    ├─ Block 4: 32×32 → 64×64
    ├─ Block 3: 64×64 → 128×128
    ├─ Block 2: 128×128 → 256×256
    └─ Block 1: 256×256 → 512×512
    ↓
Output: Saliency Map (1 channel: 0=background, 1=foreground)
```

**Encoder:** Gets smaller and smaller, learning higher-level features.  
**Decoder:** Gets bigger and bigger, reconstructing the segmentation mask.  
**Skip connections:** Connect encoder to decoder at same resolution (preserves detail).

**The output is a "saliency map":** Grayscale image where white = object, black = background.

**Training data:** Thousands of photos with hand-drawn segmentation masks.

### Why This Sometimes Fails

- **Transparent objects:** Glass, water. U2Net was trained mostly on opaque objects.
- **Objects touching background:** If a person's hand touches a wall, the boundary is ambiguous.
- **Similar colors:** White cat on white carpet.

**Our fix:** If rembg fails, we use the original image with a fake alpha channel (fully opaque).

---

## Chapter 3: Step 2 — Multi-View Generation (Zero123++)

### The Core Problem

A single photo only shows the FRONT of an object. To reconstruct 3D, we need to see it from multiple angles.

**Human analogy:** You can't sculpt a statue from one photo. You need to walk around it.

**AI solution:** Train a model that can "imagine" what the object looks like from other angles.

### What is Zero123++?

Zero123++ is a **diffusion model** that takes one image and generates 6 consistent views:

```
Original Photo (front view)
    ↓
Zero123++
    ↓
6 views arranged as a 3×2 grid:
┌─────────┬─────────┐
│  Front  │  Right  │
├─────────┼─────────┤
│   Back  │  Left   │
├─────────┼─────────┤
│   Top   │ Bottom  │
└─────────┴─────────┘
```

**Consistent means:** If the object is red in the front view, it's red in the back view too. No contradictions.

### Exercise 3.1: Understanding Diffusion

Run this mental exercise. No code needed.

**Forward diffusion (training):**
```
Real Image
    ↓ + small noise (step 1)
Slightly noisy image
    ↓ + small noise (step 2)
More noisy image
    ...
    ↓ + small noise (step 1000)
Pure noise (like TV static)
```

**Reverse diffusion (inference — what we do):**
```
Pure noise
    ↓ - predicted noise (step 1000)
Slightly structured noise
    ↓ - predicted noise (step 999)
More structured noise
    ...
    ↓ - predicted noise (step 1)
Real-looking image
```

**The model learns:** "Given this noisy image at step T, what noise was added?"

If it can predict the noise, it can subtract it. Step by step, noise becomes an image.

### Exercise 3.2: The UNet Architecture

Zero123++ uses a **UNet** with attention. Here's the simplified structure:

```python
class UNet2DConditionModel:
    """
    Input: Noisy image + timestep + conditioning (our photo)
    Output: Predicted noise
    """
    
    def forward(self, noisy_image, timestep, conditioning_image):
        # 1. Embed the timestep ("how noisy is this?")
        t_emb = timestep_embedding(timestep)
        
        # 2. Encode the conditioning image (our photo)
        #    Through a Vision Transformer (ViT)
        image_features = vit_encoder(conditioning_image)
        
        # 3. Downsample path (encoder)
        x = noisy_image
        for block in down_blocks:
            x = resnet_block(x, t_emb)
            x = attention_block(x, image_features)  # KEY: attends to our photo
            x = downsample(x)
        
        # 4. Middle block
        x = resnet_block(x, t_emb)
        x = attention_block(x, image_features)
        
        # 5. Upsample path (decoder)
        for block in up_blocks:
            x = upsample(x)
            x = resnet_block(x, t_emb)
            x = attention_block(x, image_features)
        
        # 6. Output predicted noise
        return output_conv(x)
```

**The attention mechanism is key:** At every layer, the model looks at our input photo and asks "what should I draw here, given the original image?"

**Cross-attention:** The query comes from the noisy image. The key/value come from our conditioning photo. This ensures the generated views match the original.

### Exercise 3.3: The Scheduler (How Many Steps?)

Diffusion can take 1000 steps (slow but high quality) or 50 steps (fast but lower quality).

**Our config:** 75 steps (balance)

```python
from diffusers import EulerAncestralDiscreteScheduler

scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)
```

**What the scheduler does:**
- Determines how much noise to add/remove at each step
- Handles the "noise schedule" (not linear — more noise early, less later)
- Can skip steps for faster inference (but lower quality)

**Euler Ancestral:** A fast sampler that often produces good results with 30-50 steps instead of 1000.

### Exercise 3.4: Custom UNet Loading

InstantMesh doesn't use the default Zero123++ UNet. It loads a **fine-tuned UNet**:

```python
# Load base pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="/path/to/vendored/pipeline.py"
)

# Load custom UNet weights
unet_state = torch.load("diffusion_pytorch_model.bin", map_location="cpu")
pipeline.unet.load_state_dict(unet_state, strict=True)
```

**Why a custom UNet?** The authors fine-tuned the diffusion model specifically for multi-view consistency. The base Zero123++ might generate views that don't align. The custom weights fix this.

### The Output

After ~30-50 seconds of diffusion, we get a single image:

```
Grid size: 960×640 (or similar)
Each view: 320×320
Arrangement: 3 rows × 2 columns
```

This grid is saved as `multiview.png` for debugging.

---

## Chapter 4: Step 3 — 3D Reconstruction (LRM)

### The Input

6 views (RGB images) + 6 camera poses (where the camera was for each view).

### What is a Camera Pose?

A camera pose defines where the camera is in 3D space and where it's looking.

```python
# Simplified camera pose
pose = {
    "position": [x, y, z],      # Where the camera is
    "look_at": [0, 0, 0],       # What it's looking at (the object center)
    "up": [0, 1, 0],            # Which way is up
    "fov": 30,                  # Field of view (zoom level)
}
```

For Zero123++, the 6 views are at fixed positions around the object:
- Front: (0, 0, r)
- Right: (r, 0, 0)
- Back: (0, 0, -r)
- Left: (-r, 0, 0)
- Top: (0, r, 0)
- Bottom: (0, -r, 0)

Where `r` = radius (distance from object center).

### What Does LRM Do?

**LRM = Large Reconstruction Model**

It takes 6 images + 6 camera poses and outputs a **triplane** representation of the 3D object.

### What is a Triplane?

A triplane is a clever way to represent 3D shape using three 2D images:

```
XY Plane (looking along Z axis):
┌─────────────────┐
│                 │
│   object's      │
│   shadow on     │
│   XY plane      │
│                 │
└─────────────────┘

XZ Plane (looking along Y axis):
┌─────────────────┐
│                 │
│   object's      │
│   shadow on     │
│   XZ plane      │
│                 │
└─────────────────┘

YZ Plane (looking along X axis):
┌─────────────────┐
│                 │
│   object's      │
│   shadow on     │
│   YZ plane      │
│                 │
└─────────────────┘
```

**Why three planes?** Any point in 3D space (x, y, z) can be queried by looking up:
- (x, y) on the XY plane
- (x, z) on the XZ plane
- (y, z) on the YZ plane

**Each pixel in these planes stores:**
- Density (is there material here?)
- Color (what color is it?)
- Features (learned representations)

### Exercise 4.1: How Triplanes Encode 3D

```python
import numpy as np

# Imagine a triplane with resolution 256×256
xy_plane = np.zeros((256, 256, 32))  # 32 feature channels
xz_plane = np.zeros((256, 256, 32))
yz_plane = np.zeros((256, 256, 32))

# To query a 3D point (0.5, 0.3, 0.7):
# Convert from [-1, 1] to [0, 256]
x_idx = int((0.5 + 1) / 2 * 256)
y_idx = int((0.3 + 1) / 2 * 256)
z_idx = int((0.7 + 1) / 2 * 256)

# Sample from all three planes
xy_feature = xy_plane[x_idx, y_idx, :]  # Shape: (32,)
xz_feature = xz_plane[x_idx, z_idx, :]  # Shape: (32,)
yz_feature = yz_plane[y_idx, z_idx, :]  # Shape: (32,)

# Combine (concatenate or sum)
point_feature = xy_feature + xz_feature + yz_feature

# Decode feature to density and color
density = density_decoder(point_feature)
color = color_decoder(point_feature)
```

**Key insight:** The triplane is a **compressed 3D representation**. Instead of storing a 256×256×256 voxel grid (16 million points), we store three 256×256 planes (196k points each = 590k total). Much more efficient.

### The LRM Architecture

```python
class LRM(nn.Module):
    def __init__(self):
        # Image encoder (ViT)
        self.image_encoder = VisionTransformer()
        
        # Triplane decoder
        self.triplane_decoder = TriplaneDecoder()
        
        # Volume renderer
        self.volume_renderer = VolumeRenderer()
    
    def forward_planes(self, images, cameras):
        """
        images: (B, 6, 3, H, W) — 6 views per object
        cameras: (B, 6, 16) — camera pose matrices
        """
        # 1. Encode each image
        image_features = []
        for view_idx in range(6):
            feat = self.image_encoder(images[:, view_idx])
            image_features.append(feat)
        
        # 2. Decode triplanes from image features + camera poses
        triplanes = self.triplane_decoder(image_features, cameras)
        
        return triplanes
    
    def render(self, triplanes, camera):
        """Render a view from a novel camera position."""
        # Ray marching through triplanes
        rays = generate_rays(camera)
        
        colors = []
        for ray in rays:
            # Sample points along ray
            points = sample_along_ray(ray)
            
            # Query triplane at each point
            features = query_triplane(triplanes, points)
            
            # Accumulate color (volume rendering)
            color = volume_render(features)
            colors.append(color)
        
        return torch.stack(colors)
```

**Training:** The LRM was trained on pairs of (multi-view images, 3D models). It learned to predict triplanes that, when rendered, match the input views.

**Inference (what we do):** Feed 6 views → get triplanes → extract mesh.

---

## Chapter 5: Step 4 — Mesh Extraction (FlexiCubes)

### The Problem

We have triplanes (implicit 3D representation). We need actual triangles (explicit 3D representation).

**Implicit:** "At position (0.1, 0.2, 0.3), density = 0.8"  
**Explicit:** "Triangle 1: vertices [v0, v1, v2]"

### What is Marching Cubes?

The classic algorithm for extracting surfaces from implicit fields:

```
1. Divide space into a grid of small cubes
2. For each cube, check the 8 corners:
   - If corner is INSIDE (density > threshold): mark as 1
   - If corner is OUTSIDE (density < threshold): mark as 0
3. Look up the cube configuration in a table (256 possible patterns)
4. Place triangles at the edges where inside meets outside
```

**Exercise 5.1: Marching Cubes on Paper**

Draw a 2D grid ("marching squares"). Some grid points are inside a circle, some outside. Connect the transitions with line segments. That's marching cubes in 2D.

**Problems with vanilla marching cubes:**
1. **Jagged edges:** Resolution limited by grid size
2. **Memory:** High-resolution grids need lots of memory
3. **No gradients:** Can't optimize the mesh after extraction

### What is FlexiCubes?

FlexiCubes is a **differentiable** version of marching cubes. This means:
- You can extract a mesh
- Compute a loss (how good is it?)
- Backpropagate to improve the mesh
- Iterate

**Why "differentiable" matters:** In training, the authors can optimize the mesh extraction to produce better results. At inference time, it produces higher-quality meshes than vanilla marching cubes.

### Exercise 5.2: Understanding "Differentiable"

```python
# Non-differentiable (traditional)
def marching_cubes(density_field):
    mesh = []
    for cube in grid:
        if density_at(cube.center) > threshold:
            mesh.append(triangles_for_cube(cube))
    return mesh
# Problem: "if" statement breaks gradients. Can't backpropagate.

# Differentiable (FlexiCubes)
def flexicubes(density_field):
    # Use soft thresholds (sigmoid) instead of hard "if"
    weights = sigmoid((density_field - threshold) / temperature)
    
    # Weighted sum of possible triangle configurations
    mesh = sum([weight * triangles for weight, triangles in configurations])
    
    return mesh
# Gradients flow through sigmoid and weighted sum.
```

**The "temperature" parameter:** Controls how "soft" the threshold is.
- High temperature: Smooth transitions (blurry boundaries)
- Low temperature: Sharp transitions (crisp boundaries)
- At inference: Very low temperature ≈ hard threshold

### The Extraction Process

```python
# From our code:
planes = model.forward_planes(images_t, input_cameras)
mesh_out = model.extract_mesh(
    planes,
    use_texture_map=False,
    **config.infer_config,
)

vertices, faces, vertex_colors = mesh_out
```

**What `extract_mesh` does internally:**
1. Query triplane at high-resolution grid points
2. Compute density at each point
3. Run FlexiCubes extraction
4. Post-process: remove duplicate vertices, fix winding order
5. Compute vertex colors by querying triplane color decoder

**Output:**
- `vertices`: (N, 3) array of 3D positions
- `faces`: (M, 3) array of vertex indices (each row is one triangle)
- `vertex_colors`: (N, 3) array of RGB colors

---

## Chapter 6: Step 5 — Export to GLB

### What is GLB?

GLB (GL Transmission Format Binary) is a standard for 3D assets. It's:
- **Binary:** Not human-readable, but compact
- **Single file:** Mesh + materials + textures in one file
- **Standard:** Supported by Three.js, Blender, Unity, Unreal

### GLB File Structure

```
GLB File
├── Header (20 bytes)
│   ├── Magic: "glTF"
│   ├── Version: 2
│   └── Length: total file size
├── Chunk 0: JSON
│   ├── Scene graph (nodes, meshes, materials)
│   ├── Buffer definitions
│   └── Accessors (how to read the binary data)
└── Chunk 1: Binary
    ├── Vertex positions (float32, 3 per vertex)
    ├── Vertex normals (float32, 3 per vertex)
    ├── Vertex colors (float32, 4 per vertex: RGBA)
    └── Face indices (uint16, 3 per face)
```

### Exercise 6.1: Build a GLB Manually (Almost)

```python
import numpy as np
import struct

# Create a simple triangle
vertices = np.array([
    [0.0, 0.0, 0.0],  # v0
    [1.0, 0.0, 0.0],  # v1
    [0.5, 1.0, 0.0],  # v2
], dtype=np.float32)

faces = np.array([
    [0, 1, 2],  # One triangle
], dtype=np.uint16)

vertex_colors = np.array([
    [1.0, 0.0, 0.0, 1.0],  # Red
    [0.0, 1.0, 0.0, 1.0],  # Green
    [0.0, 0.0, 1.0, 1.0],  # Blue
], dtype=np.float32)

# In reality, we'd use a library like pygltflib or trimesh
# But now you understand what goes inside the file.
```

### Our Export Code

```python
# From InstantMesh's src/utils/mesh_util.py:
# (simplified)

def save_glb(vertices, faces, vertex_colors, filepath):
    """
    vertices: (N, 3) float32
    faces: (M, 3) int32
    vertex_colors: (N, 3) float32 (RGB)
    """
    
    # Reorient to glTF coordinate system (Y-up, right-handed)
    vertices = vertices[:, [1, 2, 0]]
    
    # Create mesh using trimesh
    import trimesh
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
    )
    
    # Export as GLB
    mesh.export(filepath)
```

**Axis conversion:** InstantMesh uses a different coordinate system than glTF. We permute axes to match.

**Vertex colors vs. textures:** We use vertex colors (color per vertex) because:
1. Simpler (no UV mapping needed)
2. Good enough for coarse preview
3. Smaller file size

For production, you'd want textures (higher detail).

---

## Chapter 7: Step 6 — Refinement (Post-Processing)

After InstantMesh returns a mesh, our `mesh_processor.py` improves it:

### 7.1 Subdivision

Split each triangle into 4 smaller triangles:

```
Original triangle:
      v0
     /  \
    /    \
   v1----v2

After subdivision:
      v0
     /|\ 
    / | \
   m0-m1-m2
  /  |   |  \
 v1--m3--v2

Where m0, m1, m2, m3 are midpoints.
```

**Why:** More triangles = smoother curves, more detail.

### 7.2 Taubin Smoothing

Average each vertex position with its neighbors, but oscillate (shrink then expand) to avoid volume loss:

```
Step 1 (shrink):    v_new = v + λ * (average_of_neighbors - v)
Step 2 (expand):    v_new = v + μ * (average_of_neighbors - v)

Where λ > 0 (small), μ < 0 (negative, larger magnitude)
```

**Why not simple averaging?** Simple averaging shrinks the mesh (like a balloon deflating). Taubin smoothing preserves volume by oscillating.

### 7.3 UV Generation

If the mesh doesn't have UV coordinates, we generate spherical UVs:

```python
# For each vertex
vertex = mesh.vertices[i] - mesh.centroid  # Center at origin
normalized = vertex / np.linalg.norm(vertex)

# Spherical coordinates
u = 0.5 + arctan2(normalized[0], normalized[2]) / (2 * pi)
v = 0.5 - arcsin(normalized[1]) / pi

mesh.visual.uv[i] = [u, v]
```

**Why UVs matter:** If you want to apply a texture image later, you need UV coordinates to map pixels to vertices.

---

## Chapter 8: The Full Data Flow (Quantified)

Let's trace the data sizes at each step:

| Step | Input | Output | Time | VRAM |
|------|-------|--------|------|------|
| 1. Load image | 512×512×3 = 786 KB | 512×512×4 = 1 MB (RGBA) | 0.1s | 0 GB |
| 2. rembg | 512×512×4 | 512×512×4 | 0.5s | 0.2 GB |
| 3. Zero123++ | 512×512×3 | 960×640×3 (6 views) | 30s | 3 GB |
| 4. LRM forward | 6×320×320×3 | Triplanes: 3×256×256×32 | 5s | 2 GB |
| 5. FlexiCubes | Triplanes | Mesh: ~5000 vertices | 10s | 6 GB |
| 6. Export GLB | Vertices+faces+colors | ~500 KB file | 0.1s | 0 GB |
| **Total** | | | **~45s** | **~11 GB peak** |

**Peak VRAM usage:** ~11 GB on a 16 GB T4 GPU. That's why we offload the diffusion pipeline to CPU after step 3.

---

## Exercises for This Part

### Exercise A: Visualize a Triplane
Load a 3D model, render three orthogonal views, and arrange them as a triplane visualization.

### Exercise B: Implement Marching Squares
Write the 2D version of marching cubes. Given a 2D grid of density values, extract contour lines.

### Exercise C: Diffusion Step Simulator
Write a Python script that:
1. Loads an image
2. Adds noise progressively (forward diffusion)
3. Denoises step by step (reverse diffusion)
4. Shows the result after N steps

### Exercise D: Mesh Smoothing Comparison
Compare:
1. Simple Laplacian smoothing (averaging neighbors)
2. Taubin smoothing
3. No smoothing

Measure volume change for each.

---

**Next:** Part D — Build-From-Scratch Exercises
