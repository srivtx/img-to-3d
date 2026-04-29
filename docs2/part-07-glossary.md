# Part 7: Glossary — Every Term Explained

## A

### API (Application Programming Interface)
**Simple:** A menu of operations a program offers to other programs.
**Example:** Our API has "upload image" and "check status." The frontend calls these like a customer ordering from a menu.

### ASGI (Asynchronous Server Gateway Interface)
**Simple:** A standard that lets Python web servers handle many requests at once without waiting.
**Example:** Uvicorn uses ASGI to run FastAPI. While one request waits for AI, another request can be processed.

### Async / Asynchronous
**Simple:** Doing multiple things without waiting for each to finish before starting the next.
**Example:** You can start cooking pasta AND chop vegetables at the same time. Async programming lets a server do the same.

### Attention (in Transformers)
**Simple:** A mechanism that lets the model focus on important parts of input.
**Example:** When reading "The cat sat on the mat because it was tired," attention helps the model connect "it" to "cat."

## B

### Background Task
**Simple:** Work that happens after the server has already responded.
**Example:** You upload a photo. The server immediately says "got it!" Then, in the background, it processes the image for 30 seconds.

### Buffer (Output Buffer)
**Simple:** A temporary holding area for data before it's written out.
**Example:** Python's `print()` collects text in a buffer. It only actually prints when the buffer fills or you force it.

## C

### Checkpoint
**Simple:** A file containing all the learned numbers (weights) of a trained model.
**Example:** `instant_mesh_large.ckpt` is 2 GB of numbers that represent everything InstantMesh learned during training.

### CORS (Cross-Origin Resource Sharing)
**Simple:** A browser security rule about which websites can talk to which servers.
**Example:** If your frontend is at `site-a.com` and your API is at `site-b.com`, the browser blocks it unless the API says "it's okay."

### CUDA
**Simple:** NVIDIA's language for making GPUs do general computing (not just graphics).
**Example:** PyTorch uses CUDA to run neural networks on NVIDIA GPUs.

### CUDA OOM (Out Of Memory)
**Simple:** The GPU ran out of memory (VRAM) and couldn't allocate more.
**Example:** Trying to load a 20 GB model on a 16 GB GPU causes CUDA OOM.

## D

### Diffusion
**Simple:** An AI technique that generates images by gradually removing noise from random static.
**Example:** Start with TV static. The model slowly "denoises" it into a photo of a cat. This takes 50-1000 steps.

### Diffusers
**Simple:** A Hugging Face library that makes diffusion models easy to use.
**Example:** We use Diffusers to load the Zero123++ pipeline.

### Differentiable
**Simple:** A function that has a well-defined slope everywhere, allowing gradient-based optimization.
**Example:** Mesh extraction via DMTet is "differentiable" — you can optimize the mesh directly using gradients.

### Docker
**Simple:** A tool that packages your application with everything it needs to run, so it works the same everywhere.
**Example:** Our Dockerfile tells Hugging Face Spaces exactly how to build and run our app.

### DOM (Document Object Model)
**Simple:** The browser's internal representation of a web page as a tree of objects.
**Example:** JavaScript uses the DOM to find elements (like `document.getElementById('uploadZone')`) and modify them.

## E

### Embedding
**Simple:** A list of numbers that represents something (word, image, concept) in a way machines understand.
**Example:** The photo you upload becomes a 512-number embedding that captures "what's in this image."

### Epoch
**Simple:** One complete pass through the training dataset.
**Example:** If you have 10,000 training images, one epoch = the model sees all 10,000 once.

## F

### Face (in 3D)
**Simple:** A triangle defined by 3 vertices.
**Example:** A cube has 12 faces (each square side is 2 triangles).

### Feed-Forward
**Simple:** A neural network that processes input in one pass, without loops or iteration.
**Example:** InstantMesh is feed-forward: photo → model → mesh. No optimization loop.

### FlexiCubes
**Simple:** A method for extracting meshes from neural fields that's better than traditional marching cubes.
**Example:** InstantMesh uses FlexiCubes to convert its internal representation into actual triangles.

### FP16 (Half Precision)
**Simple:** Using 16-bit numbers instead of 32-bit for model weights. Faster and half the memory.
**Example:** A 4 GB model becomes 2 GB in FP16. Slightly lower precision, usually not noticeable.

## G

### Gaussian Splatting
**Simple:** A 3D rendering technique using millions of fuzzy blobs instead of triangles.
**Example:** Each "blob" is a 3D Gaussian with position, color, and size. Rendered together, they look photorealistic.

### GLB / glTF
**Simple:** A standard file format for 3D scenes. GLB is the binary (single-file) version.
**Example:** Browsers can load GLB files directly into Three.js.

### GPU (Graphics Processing Unit)
**Simple:** A processor with thousands of small cores, great at doing the same math operation on many numbers at once.
**Example:** Training or running AI models is 10-100x faster on GPU than CPU.

### Gradient
**Simple:** The slope of a function. Used in training to know which direction to adjust weights.
**Example:** If the gradient is positive, decrease the weight. If negative, increase it.

## H

### HTTP
**Simple:** The language browsers and servers use to communicate.
**Example:** `GET /jobs/abc123` means "please tell me the status of job abc123."

### Hugging Face
**Simple:** A platform that hosts AI models, datasets, and tools.
**Example:** We download InstantMesh weights from Hugging Face.

## I

### Icosphere
**Simple:** A sphere made of triangles (like a geodesic dome).
**Example:** Our mock mode generates an icosphere when the real AI fails to load.

### Implicit Field
**Simple:** A neural network that answers "what's inside/outside?" instead of storing explicit triangles.
**Example:** The network says "at position (0.5, 0.2, 0.1), the value is -0.3" (negative = inside the object).

### Inference
**Simple:** Using a trained model to make predictions on new data.
**Example:** Uploading a photo and getting a 3D mesh is inference. We're not training anything.

## J

### JSON
**Simple:** A text format for structured data. Looks like Python dictionaries.
**Example:** `{"status": "completed", "progress": 100}`

### Job (in our system)
**Simple:** A unit of work tracked in our queue. Has an ID, status, and progress.
**Example:** When you upload a photo, we create a job. You poll for its status.

## K

### Kernel (CUDA)
**Simple:** A function that runs on the GPU.
**Example:** A CUDA kernel might multiply two matrices. Thousands of threads run it in parallel.

## L

### Latent Space
**Simple:** A compressed representation where similar things are close together.
**Example:** In latent space, a photo of a cat and a drawing of a cat are near each other.

### Loss Function
**Simple:** A score that tells the model how wrong its prediction is.
**Example:** If the model predicts "dog" but the answer is "cat," the loss is high. Training minimizes loss.

### LRM (Large Reconstruction Model)
**Simple:** A neural network that reconstructs 3D from multiple 2D views.
**Example:** InstantMesh's LRM takes 6 views and outputs a 3D representation.

## M

### Marching Cubes
**Simple:** An algorithm that turns an implicit field (inside/outside values) into triangles.
**Example:** Like a 3D contour plot. It finds where the surface crosses from inside to outside.

### Material (3D)
**Simple:** Properties that define how light interacts with a surface.
**Example:** Shiny metal vs. dull plastic. Defined by color, roughness, metalness.

### Mesh
**Simple:** A collection of vertices and faces that defines a 3D shape.
**Example:** A character in a video game is a mesh with 50,000 triangles.

### Mock
**Simple:** A fake implementation used for testing when the real thing isn't available.
**Example:** Our mock mode returns a sphere instead of running AI.

### Model (AI)
**Simple:** A neural network with trained weights. The "brain" that makes predictions.
**Example:** InstantMesh is a model. It takes photos and outputs 3D.

### Multi-View
**Simple:** Multiple photographs of the same object from different angles.
**Example:** Front, back, left, right, top, bottom = 6 views.

## N

### NeRF (Neural Radiance Field)
**Simple:** A neural network that represents a 3D scene. It answers "what color is visible from here?"
**Example:** Instead of storing triangles, it stores a network. Render by asking the network about many camera positions.

### Normal (3D)
**Simple:** A direction vector perpendicular to a surface. Used for lighting.
**Example:** The normal of a floor points straight up. Light hitting at a shallow angle creates long shadows.

### Normalization
**Simple:** Scaling data to a standard range (usually 0-1 or -1 to 1).
**Example:** Pixel values 0-255 become 0.0-1.0 before feeding to a neural network.

## O

### Objective-C / Objective Function
**Simple:** See "Loss Function."

### ONNX
**Simple:** A standard format for machine learning models.
**Example:** `rembg` uses an ONNX model for background removal.

### ONNX Runtime
**Simple:** A library that runs ONNX models efficiently.
**Example:** Missing this causes `import rembg` to fail.

### Optimization (in ML)
**Simple:** Iteratively adjusting model weights to reduce loss.
**Example:** Gradient descent is an optimization algorithm.

## P

### Parameter
**Simple:** A number inside a neural network that gets adjusted during training.
**Example:** A model with 100 million parameters has 100 million numbers that were learned from data.

### Pipeline (in Diffusers)
**Simple:** A pre-packaged workflow that combines multiple models.
**Example:** The Zero123++ pipeline includes a VAE, UNet, and scheduler.

### Polling
**Simple:** Repeatedly asking for updates.
**Example:** Our frontend asks "is it done yet?" every second.

### Port
**Simple:** A door number for network connections.
**Example:** Our server listens on port 8000.

### Pre-trained
**Simple:** A model that was already trained on a large dataset.
**Example:** We download pre-trained InstantMesh weights. We don't train from scratch.

## Q

### Quantization
**Simple:** Using fewer bits to store model weights. Faster but slightly less accurate.
**Example:** FP32 → INT8 reduces size by 4x. Good for edge deployment.

### Queue
**Simple:** A line of jobs waiting to be processed.
**Example:** Our in-memory queue tracks which uploads are pending, running, or done.

## R

### Reconstruction
**Simple:** Creating 3D geometry from 2D images.
**Example:** InstantMesh's reconstruction stage turns 6 views into a mesh.

### Refinement
**Simple:** Improving a coarse result with additional processing.
**Example:** Our Trimesh refinement smooths and subdivides the coarse mesh.

### REST
**Simple:** A style of designing APIs using standard HTTP methods.
**Example:** `GET` to read, `POST` to create, `DELETE` to remove.

### RGB / RGBA
**Simple:** Color formats. RGB = red, green, blue. RGBA adds alpha (transparency).
**Example:** PNGs support RGBA. JPEGs only support RGB.

## S

### Scheduler (in Diffusers)
**Simple:** Controls how much noise is removed at each diffusion step.
**Example:** Euler ancestral scheduler. Different schedulers affect speed vs quality.

### Score Distillation Sampling (SDS)
**Simple:** A technique that uses a diffusion model to guide 3D optimization.
**Example:** DreamFusion uses SDS to optimize a NeRF until rendered views match the diffusion model's idea of the object.

### Shim
**Simple:** A small piece of code that fills a gap or fixes incompatibility.
**Example:** Our transformers shim re-adds functions that were removed in version 5.x.

### Singleton
**Simple:** A design pattern where only one instance of a class exists.
**Example:** Our `CoarseGenerator` is a singleton. Only one model is loaded, shared by all requests.

### Static File
**Simple:** A file that doesn't change (CSS, JS, images).
**Example:** `style.css` is static. The server just reads it from disk and sends it.

### Subdivision
**Simple:** Adding more triangles to a mesh by splitting existing ones.
**Example:** A triangle becomes 4 smaller triangles. Smooths curves.

### Subprocess
**Simple:** Running another program from within your program.
**Example:** Starting Uvicorn as a subprocess from a Colab notebook.

### Supervised Learning
**Simple:** Training with labeled examples (input + correct output).
**Example:** Show the model 10,000 photos labeled "cat" or "dog."

## T

### Taubin Smoothing
**Simple:** A mesh smoothing algorithm that preserves shape better than simple averaging.
**Example:** Our refinement uses Taubin smoothing to reduce jagged edges.

### Tensor
**Simple:** A multi-dimensional array of numbers. The basic data structure in ML.
**Example:** A 224×224 RGB image is a tensor of shape [3, 224, 224].

### Three.js
**Simple:** A JavaScript library for rendering 3D in browsers.
**Example:** Our frontend uses Three.js to display GLB files.

### Token
**Simple:** A unit of text that a language model processes (word, part of word, or punctuation).
**Example:** "Hello world!" might be 3 tokens: ["Hello", " world", "!"]

### Transformers (library)
**Simple:** Hugging Face's library for pre-trained language and vision models.
**Example:** InstantMesh uses Transformers for its ViT encoder.

### Transformer (architecture)
**Simple:** A neural network architecture using attention mechanisms.
**Example:** GPT, BERT, and ViT are all transformer architectures.

### Trimesh
**Simple:** A Python library for loading, manipulating, and saving 3D meshes.
**Example:** We use Trimesh for refinement (subdivide, smooth, export GLB).

### Triplane
**Simple:** A 3D representation using three 2D feature planes (XY, XZ, YZ).
**Example:** InstantMesh's LRM outputs a triplane that encodes 3D shape.

### Tunnel
**Simple:** A public URL that forwards to a local server.
**Example:** Cloudflare tunnel gives us `https://abc.trycloudflare.com` pointing to `localhost:8000`.

## U

### UNet
**Simple:** A neural network architecture shaped like a U. Used in diffusion models.
**Example:** The diffusion model's UNet predicts noise to remove at each step.

### UV Coordinates
**Simple:** Mapping from 3D vertices to 2D texture positions.
**Example:** Like unfolding a cardboard box. The UV map is the flat pattern.

## V

### Vertex
**Simple:** A point in 3D space.
**Example:** A triangle has 3 vertices. Each has x, y, z coordinates.

### Vertex Colors
**Simple:** Colors stored directly on vertices (not using a texture image).
**Example:** If vertex A is red and vertex B is blue, the GPU blends them across the face.

### ViT (Vision Transformer)
**Simple:** A transformer that processes images by splitting them into patches.
**Example:** InstantMesh uses a ViT to understand the input photo.

### VRAM (Video RAM)
**Simple:** Memory on the GPU.
**Example:** T4 GPU has 16 GB VRAM. Models must fit in this space.

## W

### WebGL
**Simple:** A browser API for rendering 3D graphics.
**Example:** Three.js uses WebGL under the hood to draw triangles on the canvas.

### WebSocket
**Simple:** A persistent connection between browser and server for real-time communication.
**Example:** Instead of polling every second, the server could push updates via WebSocket.

### Weight
**Simple:** A number in a neural network that's learned during training.
**Example:** A model with 100 million weights has 100 million learned numbers.

## Z

### Zero123 / Zero123++
**Simple:** Models that generate novel views of an object from a single image.
**Example:** Show Zero123 a photo of a chair, and it generates what the chair looks like from the back.

### Zero-Shot
**Simple:** Performing a task without task-specific training.
**Example:** InstantMesh generates 3D from any photo without retraining.

---

## Quick Reference: Abbreviations

| Abbreviation | Full Name | What It Is |
|--------------|-----------|------------|
| AI | Artificial Intelligence | Computers doing tasks that seem intelligent |
| API | Application Programming Interface | How programs talk to each other |
| ASGI | Asynchronous Server Gateway Interface | Python web server standard |
| CKPT | Checkpoint | Saved model weights |
| CUDA | Compute Unified Device Architecture | NVIDIA's GPU computing platform |
| DOM | Document Object Model | Browser's page representation |
| FP16 | 16-bit Floating Point | Half-precision numbers |
| GLB | GL Transmission Format Binary | 3D file format |
| GPU | Graphics Processing Unit | Parallel processor |
| HF | Hugging Face | AI model hosting platform |
| HTTP | HyperText Transfer Protocol | Web communication standard |
| JSON | JavaScript Object Notation | Data format |
| LRM | Large Reconstruction Model | 3D reconstruction network |
| ML | Machine Learning | Computers learning from data |
| NeRF | Neural Radiance Field | Neural 3D scene representation |
| OOM | Out Of Memory | Ran out of RAM/VRAM |
| ONNX | Open Neural Network Exchange | Model format standard |
| REST | Representational State Transfer | API design style |
| RGB | Red Green Blue | Color format |
| RGBA | Red Green Blue Alpha | Color + transparency |
| SDS | Score Distillation Sampling | 3D optimization technique |
| T4 | Tesla T4 | NVIDIA GPU model (16 GB VRAM) |
| UV | Texture coordinates | 2D mapping for 3D textures |
| ViT | Vision Transformer | Image-processing transformer |
| VRAM | Video RAM | GPU memory |
| WebGL | Web Graphics Library | Browser 3D API |

---

## End of docs2/

Thank you for reading. If you made it this far, you now understand:
- What 3D models actually are (triangles!)
- What AI does (pattern matching at scale)
- How the web works (requests and responses)
- What we built (layer by layer)
- What broke (everything, then we fixed it)
- What else is possible (20+ architectures)
- What every term means

**The most important lesson:** Build one layer at a time. Make it work. Then add the next layer. Complexity is earned, not given.

---

*Written with care for beginners who will one day be experts.*
