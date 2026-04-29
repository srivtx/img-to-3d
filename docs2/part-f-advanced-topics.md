# Part F: Advanced Topics

## Introduction

This section covers topics that are beyond the MVP but important for production, research, and understanding the cutting edge.

Each topic includes:
- What it is
- Why it matters
- How it applies to our project
- Code examples where relevant

---

## Topic 1: Model Quantization

### What is Quantization?

Neural networks use 32-bit floating point numbers (FP32) by default. Quantization converts them to lower precision:

| Format | Bits | Memory | Speed | Accuracy |
|--------|------|--------|-------|----------|
| FP32 | 32 | 100% | Baseline | 100% |
| FP16 | 16 | 50% | ~2x | ~99% |
| INT8 | 8 | 25% | ~4x | ~95% |
| INT4 | 4 | 12.5% | ~8x | ~90% |

### Why It Matters

Our model uses ~12 GB VRAM at FP16. At INT8, it would use ~6 GB. This means:
- Smaller GPUs can run it
- More concurrent jobs per GPU
- Faster inference

### How It Works

**Post-Training Quantization:**
```python
import torch

# Load model
model = torch.load("model.ckpt")

# Quantize to INT8
quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save
torch.save(quantized, "model_int8.ckpt")
```

**The trick:** Find the min/max range of weights in each layer. Map FP32 values to INT8 using:
```
scale = (max - min) / 255
quantized = round((fp32 - min) / scale)
fp32_approx = quantized * scale + min
```

**Loss of precision:** Values are rounded. But neural networks are robust to small perturbations.

### Quantization-Aware Training (Better but Harder)

Instead of quantizing after training, simulate quantization during training:
```python
# During forward pass
fake_quantized_weight = quantize(weight) + dequantize(quantize(weight))
# Model learns to be robust to quantization
```

### Our Application

**Current:** We use FP16 on CUDA, FP32 on CPU/MPS.

**Future:** Experiment with INT8 for:
- CPU inference (4x speedup)
- Edge deployment (mobile devices)
- Higher throughput on server GPUs

**Tools:**
- PyTorch quantization (built-in)
- ONNX Runtime (optimized inference)
- TensorRT (NVIDIA-specific, fastest)

---

## Topic 2: Model Distillation

### What is Distillation?

Train a small model (student) to mimic a large model (teacher).

```
Large Model (Teacher)
    Input: Photo of cat
    Output: [0.9, 0.05, 0.05]  (cat, dog, bird)
    Parameters: 100M
    
Small Model (Student)
    Input: Photo of cat  
    Output: [0.85, 0.08, 0.07]  (close to teacher)
    Parameters: 10M
```

**The student learns the teacher's "soft targets"** (probabilities) not just hard labels.

### Why It Matters

- **Speed:** 10M model runs 10x faster than 100M model
- **Memory:** Fits on smaller devices
- **Cost:** Cheaper to deploy

### How It Works

```python
# Training loop
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0):
    # Soft targets from teacher (with temperature)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    
    # Student predictions (with same temperature)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    
    # KL divergence loss
    distill_loss = F.kl_div(student_soft, soft_targets, reduction='batchmean')
    
    # Hard label loss (standard cross-entropy)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combined
    return alpha * distill_loss + (1 - alpha) * hard_loss
```

**Temperature:** Higher temperature (>1) makes probability distributions softer, revealing more information about the teacher's confidence.

### Our Application

**Future work:** Distill InstantMesh into a smaller model:
- Target: 10x faster, 50% of VRAM
- Use for real-time preview
- Keep full model for final quality

---

## Topic 3: Knowledge Distillation for 3D

### Specific Challenges

3D models have unique outputs:
- Multi-view images (not just classification)
- Triplane features (not just vectors)
- Mesh geometry (structured output)

**Distillation strategies:**
1. **Feature distillation:** Match intermediate representations (triplanes)
2. **Render distillation:** Match rendered views from different angles
3. **Mesh distillation:** Match vertex positions directly

```python
# Render distillation
teacher_views = teacher_render(mesh, cameras)
student_views = student_render(mesh, cameras)
loss = mse_loss(student_views, teacher_views)
```

---

## Topic 4: Continuous Learning / Fine-Tuning

### What is Fine-Tuning?

Take a pre-trained model and train it further on your specific data.

**Example:** InstantMesh was trained on general objects. Fine-tune it on shoes for a shoe e-commerce app.

### How to Fine-Tune

```python
# Load pre-trained model
model = load_instantmesh()

# Freeze early layers (they learn general features)
for param in model.encoder.parameters():
    param.requires_grad = False

# Only train the decoder
optimizer = Adam(model.decoder.parameters(), lr=1e-4)

# Train on your data
for batch in dataloader:
    images, meshes = batch
    predicted = model(images)
    loss = mesh_loss(predicted, meshes)
    loss.backward()
    optimizer.step()
```

**Why freeze early layers?** They learned general features (edges, textures) that apply to all images. Only the later layers need to adapt to your domain.

### Our Application

**Potential fine-tuning domains:**
- Furniture (IKEA-style models)
- Shoes/fashion
- Architecture/buildings
- Vehicles

**Requirement:** Need paired data (photos + 3D models). Expensive to collect.

---

## Topic 5: Test-Time Optimization (TTO)

### What is TTO?

Instead of running the model once, optimize the output for a specific input.

**Standard inference:**
```
Photo → Model → Mesh (done)
```

**Test-time optimization:**
```
Photo → Model → Initial Mesh
                 ↓
            Render views
                 ↓
            Compare to photo
                 ↓
            Adjust mesh
                 ↓
            (repeat 100 times)
                 ↓
            Refined Mesh
```

### Why It Helps

The model gives a good initial guess. TTO refines it for the specific photo.

**Trade-off:** 2-5 minutes of optimization vs. 30 seconds of one-shot inference.

### How It Works

```python
mesh = model.generate_mesh(photo)  # Initial guess

for step in range(1000):
    # Render mesh from camera angle
    rendered = render(mesh, camera)
    
    # Compare to target photo
    loss = mse_loss(rendered, target_photo)
    
    # Backpropagate to mesh vertices
    loss.backward()
    
    # Update vertex positions
    optimizer.step()
```

**This is how DreamFusion works.** It uses a diffusion model's score function as the loss.

### Our Application

**Future feature:** "Refine" button that runs 2 minutes of TTO for higher quality.

---

## Topic 6: Neural Radiance Fields (NeRF)

### What is NeRF?

NeRF represents a 3D scene as a neural network, not as triangles.

**Traditional 3D:** Store vertices and faces.  
**NeRF:** Store a neural network that answers: "What color is visible from position (x,y,z) looking direction (dx,dy,dz)?"

### How It Works

```python
class NeRF(nn.Module):
    def forward(self, position, direction):
        # Position encoding (crucial for high-frequency detail)
        pos_encoded = positional_encoding(position, L=10)
        dir_encoded = positional_encoding(direction, L=4)
        
        # MLP
        x = self.mlp(pos_encoded)
        density = x[..., 0]  # How much "stuff" is here?
        feature = x[..., 1:]
        
        # Color depends on direction (view-dependent effects)
        color = self.color_head(torch.cat([feature, dir_encoded], dim=-1))
        
        return density, color
```

**Positional encoding:** Map positions to high-frequency space using sin/cos:
```python
def positional_encoding(x, L):
    encoded = [x]
    for i in range(L):
        encoded.append(torch.sin(2**i * np.pi * x))
        encoded.append(torch.cos(2**i * np.pi * x))
    return torch.cat(encoded, dim=-1)
```

**Why positional encoding?** Neural networks struggle to learn high-frequency details (sharp edges). Encoding helps them represent fine structure.

### Volume Rendering

To render an image from a NeRF:

```python
def render_ray(nerf, ray_origin, ray_direction):
    # Sample points along the ray
    points = sample_points(ray_origin, ray_direction, near=2.0, far=6.0, n_samples=64)
    
    # Query NeRF at each point
    densities, colors = nerf(points, ray_direction)
    
    # Volume rendering (accumulate color along ray)
    rendered_color = volume_render(densities, colors)
    
    return rendered_color
```

**Volume rendering equation:**
```
Color = Σ [transmittance_i × density_i × color_i]
```

Where `transmittance` = probability that light reaches this point without being blocked.

### NeRF vs. Mesh

| Aspect | NeRF | Mesh |
|--------|------|------|
| Representation | Neural network | Triangles |
| Memory | ~5-50 MB | ~1-50 MB |
| Rendering | Slow (~1s/frame) | Fast (~60 fps) |
| Quality | Photorealistic | Geometric |
| Editing | Hard | Easy |
| Animation | Hard | Easy |

### Our Application

**Future output option:** Generate NeRF in addition to mesh. Users who want photorealistic rendering can use the NeRF. Users who want to edit/animate can use the mesh.

---

## Topic 7: 3D Gaussian Splatting

### What is It?

Instead of triangles or neural networks, represent the scene as **millions of 3D Gaussians** (fuzzy blobs).

### A Gaussian in 3D

```python
def gaussian_3d(position, mean, covariance, opacity):
    """
    position: (x, y, z) where we evaluate
    mean: center of the Gaussian
    covariance: shape (ellipsoid)
    opacity: how opaque (0-1)
    """
    diff = position - mean
    exponent = -0.5 * diff.T @ np.linalg.inv(covariance) @ diff
    value = opacity * np.exp(exponent)
    return value
```

**Parameters per Gaussian:**
- Position (3)
- Covariance (3×3 symmetric = 6 unique values)
- Color (3)
- Opacity (1)
- **Total: 13 parameters per Gaussian**

### Rendering

```python
def render_gaussians(camera, gaussians):
    # Project Gaussians to 2D (screen space)
    gaussians_2d = project_to_screen(gaussians, camera)
    
    # Sort by depth (back to front)
    gaussians_2d = sort_by_depth(gaussians_2d)
    
    # Splat onto image (alpha compositing)
    image = np.zeros((H, W, 3))
    for gaussian in gaussians_2d:
        image = alpha_composite(image, gaussian)
    
    return image
```

**Speed:** Millions of Gaussians render in real-time using CUDA.

### Training

1. Start with random Gaussians or point cloud
2. Render and compare to photos
3. Adjust Gaussian parameters (gradient descent)
4. Split large Gaussians, prune transparent ones
5. Repeat

### Our Application

**Future feature:**
- Train Gaussian Splatting from InstantMesh output
- Real-time viewer with photorealistic quality
- Export as `.ply` or `.splat` file

---

## Topic 8: Diffusion Model Internals (Deep Dive)

### The UNet Architecture

```python
class UNet2DConditionModel:
    def __init__(self):
        # Time embedding (tells model which diffusion step)
        self.time_embed = TimestepEmbedding(320)
        
        # Down blocks (encoder)
        self.down_blocks = nn.ModuleList([
            CrossAttnDownBlock2D(in_channels=4, out_channels=320),
            CrossAttnDownBlock2D(in_channels=320, out_channels=640),
            CrossAttnDownBlock2D(in_channels=640, out_channels=1280),
        ])
        
        # Middle block
        self.mid_block = UNetMidBlock2DCrossAttn(in_channels=1280)
        
        # Up blocks (decoder)
        self.up_blocks = nn.ModuleList([
            CrossAttnUpBlock2D(in_channels=1280, out_channels=1280),
            CrossAttnUpBlock2D(in_channels=1280, out_channels=640),
            CrossAttnUpBlock2D(in_channels=640, out_channels=320),
        ])
        
        # Output
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3)
    
    def forward(self, noisy_latent, timestep, encoder_hidden_states):
        # 1. Time embedding
        t_emb = self.time_embed(timestep)
        
        # 2. Initial convolution
        sample = self.conv_in(noisy_latent)
        
        # 3. Down blocks
        down_block_res_samples = []
        for downblock in self.down_blocks:
            sample, res_samples = downblock(sample, t_emb, encoder_hidden_states)
            down_block_res_samples.extend(res_samples)
        
        # 4. Middle block
        sample = self.mid_block(sample, t_emb, encoder_hidden_states)
        
        # 5. Up blocks
        for upblock in self.up_blocks:
            res_samples = down_block_res_samples[-len(upblock.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upblock.resnets)]
            sample = upblock(sample, res_samples, t_emb, encoder_hidden_states)
        
        # 6. Output
        sample = self.conv_out(sample)
        return sample
```

**Key components:**
- **ResNet blocks:** Process features at each resolution
- **Attention blocks:** Allow model to focus on relevant parts
- **Cross-attention:** Attend to text/image conditioning
- **Skip connections:** Connect encoder to decoder (preserves detail)

### The Diffusion Process

```python
def forward_diffusion(x_0, t, noise_schedule):
    """Add noise to clean image x_0 at timestep t."""
    # Get noise level for timestep t
    alpha_bar = noise_schedule.alpha_bars[t]
    
    # Sample random noise
    noise = torch.randn_like(x_0)
    
    # Mix image and noise
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
    
    return x_t, noise

def reverse_diffusion_step(model, x_t, t, noise_schedule):
    """One step of denoising."""
    # Predict noise
    predicted_noise = model(x_t, t, conditioning)
    
    # Compute x_{t-1} from x_t and predicted noise
    alpha = noise_schedule.alphas[t]
    alpha_bar = noise_schedule.alpha_bars[t]
    alpha_bar_prev = noise_schedule.alpha_bars[t-1]
    
    # Formula depends on scheduler (DDPM, DDIM, Euler, etc.)
    # Simplified:
    x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
    x_t_prev = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_prev) * predicted_noise
    
    return x_t_prev
```

### Schedulers

Different schedulers trade speed vs quality:

| Scheduler | Steps | Quality | Speed | Best For |
|-----------|-------|---------|-------|----------|
| DDPM | 1000 | Best | Slowest | Training |
| DDIM | 50-100 | Good | Fast | Inference |
| Euler | 20-50 | Good | Fast | Inference |
| DPM++ 2M | 20-30 | Excellent | Fast | Inference |
| UniPC | 15-25 | Excellent | Fastest | Inference |

**Our choice:** Euler Ancestral with 75 steps (balance of quality and speed).

---

## Topic 9: Ethical Considerations

### Copyright and Training Data

**Question:** If InstantMesh was trained on copyrighted images, are its outputs derivative works?

**Current legal status (unclear):**
- Models learn patterns, not specific images
- Output is generated, not copied
- But training on copyrighted data without license is contested

**Mitigation:**
- Use models with clear licenses (Apache 2.0, MIT)
- Document data sources
- Consider fine-tuning on licensed data

### Deepfakes and Misuse

**Risk:** 3D models of real people without consent.

**Mitigation:**
- Terms of service prohibiting non-consensual use
- Watermarking outputs
- Reporting mechanisms

### Environmental Impact

**Training one large model:** ~500 MWh of electricity = ~50 tons CO2

**Inference (one generation):** ~0.5 kWh = negligible

**Our impact:** Small. We use pre-trained models. Inference only.

---

## Topic 10: The Future of Image-to-3D

### Near-Term (1-2 years)
- **Real-time generation:** <1 second on consumer hardware
- **Higher quality:** Details like wrinkles, fabric texture
- **Text-guided editing:** "Make this chair red and taller"
- **Video input:** Generate 3D from a short video clip

### Medium-Term (3-5 years)
- **Physical accuracy:** Correct mass, center of gravity, material properties
- **Animation-ready:** Automatic rigging and skinning
- **Multi-object scenes:** Entire rooms, not just single objects
- **Style transfer:** "Render this in Pixar style"

### Long-Term (5-10 years)
- **Neural rendering:** No explicit geometry, just neural fields
- **Holographic displays:** 3D without glasses
- **Physical fabrication:** Direct to 3D printer with material properties
- **World models:** AI that understands 3D space like humans do

---

## Summary

| Topic | What It Is | Why It Matters |
|-------|-----------|----------------|
| **Quantization** | Lower precision weights | Speed, memory, edge deployment |
| **Distillation** | Small model mimics large | Real-time inference |
| **Fine-tuning** | Adapt model to domain | Better quality for specific use cases |
| **TTO** | Optimize per input | Higher quality, slower |
| **NeRF** | Neural 3D representation | Photorealistic rendering |
| **Gaussians** | Fuzzy blobs for 3D | Real-time, photorealistic |
| **Diffusion internals** | How denoising works | Understanding, optimization |
| **Ethics** | Legal, social impacts | Responsible deployment |

---

**End of docs2/**

Thank you for reading. You now have a ground-up understanding of image-to-3D generation, from Python sockets to neural radiance fields.

**Remember:** Build one layer at a time. Make it work. Then add the next layer.

---

*Written for beginners who will build the future.*
