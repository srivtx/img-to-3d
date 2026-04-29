# Google Colab Notebook

Open this notebook in Google Colab to run Image-to-3D on free T4 GPU.

## Quick Start

1. Download [`Image_to_3D_Generator.ipynb`](Image_to_3D_Generator.ipynb)
2. Go to [colab.research.google.com](https://colab.research.google.com)
3. File → Upload notebook
4. Runtime → Change runtime type → GPU
5. Run all cells (Ctrl+F9)
6. Wait for the public URL to appear

## What Happens

The notebook will:
1. Clone the GitHub repo
2. Install PyTorch, FastAPI, Three.js dependencies
3. Download InstantMesh AI weights (~4GB, one-time)
4. Start the FastAPI server
5. Create a public tunnel via cloudflared
6. Give you a clickable URL

## Requirements

- Google account (free)
- GPU runtime enabled (T4 is free)
- ~6GB disk space in Colab

## Limitations

- Colab session disconnects after ~12 hours or idle timeout
- Tunnel URL changes every restart
- For persistent deployment, use Hugging Face Spaces or RunPod

## Troubleshooting

**"No GPU found"** → Make sure Runtime → Change runtime type → GPU is selected

**"CUDA out of memory"** → Runtime → Restart runtime, then run all cells again

**"Tunnel URL not appearing"** → Wait 30-60 seconds, or check the debug cell at the bottom
