"""FastAPI application entrypoint."""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

from app.core.config import OUTPUTS_DIR, DEVICE, KEEP_MODELS_IN_MEMORY, JOB_TTL_SECONDS
from app.core.models import Generate3DResponse, JobStatusResponse, HealthResponse, JobStatus
from app.services.queue import queue
from app.pipeline.instantmesh import coarse_generator
from app.pipeline.refinement import refinement_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


async def cleanup_worker():
    """Background task to clean old jobs."""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        await queue.cleanup_old_jobs(JOB_TTL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    # Preload models if configured
    if KEEP_MODELS_IN_MEMORY:
        coarse_generator.load()
    
    # Start cleanup worker
    task = asyncio.create_task(cleanup_worker())
    yield
    task.cancel()


app = FastAPI(
    title="Fast Image-to-3D Generation Server",
    description="Progressive coarse→fine 3D generation API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output files and frontend statically
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the main frontend application."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, detail="Frontend not built")
    return FileResponse(str(index_path))


async def process_job(job_id: str, image_path: str):
    """Background task: coarse generation → refinement."""
    try:
        # --- Phase 1: Coarse generation ---
        await queue.update_job(
            job_id,
            status=JobStatus.PROCESSING_COARSE,
            progress_percent=10,
            message="Generating coarse mesh...",
        )
        
        output_dir = os.path.join(str(OUTPUTS_DIR), job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        preview_path, _ = coarse_generator.generate(image_path, output_dir)
        preview_url = f"/outputs/{job_id}/preview.glb"
        
        await queue.update_job(
            job_id,
            status=JobStatus.COARSE_READY,
            preview_path=preview_path,
            progress_percent=40,
            message="Preview ready, starting refinement...",
        )
        
        # --- Phase 2: Refinement ---
        await queue.update_job(
            job_id,
            status=JobStatus.REFINING,
            progress_percent=50,
            message="Refining mesh and textures...",
        )
        
        final_path = refinement_pipeline.refine(preview_path, output_dir)
        final_url = f"/outputs/{job_id}/final.glb"
        
        await queue.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            final_path=final_path,
            progress_percent=100,
            message="Complete! High-quality model ready.",
        )
        
    except Exception as e:
        await queue.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=str(e),
            message=f"Processing failed: {str(e)}",
        )


@app.post("/generate-3d", response_model=Generate3DResponse)
async def generate_3d(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
):
    """
    Upload an image and start 3D generation.
    Returns immediately with preview URL (once ready) and job ID for polling.
    """
    # Validate
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, detail="Invalid image type. Use JPEG, PNG, or WebP.")
    
    # Save uploaded file
    job = await queue.create_job(image_path="")
    upload_dir = os.path.join(str(OUTPUTS_DIR), job.job_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    image_path = os.path.join(upload_dir, f"input_{image.filename}")
    with open(image_path, "wb") as f:
        f.write(await image.read())
    
    await queue.update_job(job.job_id, image_path=image_path)
    
    # Kick off background processing
    background_tasks.add_task(process_job, job.job_id, image_path)
    
    return Generate3DResponse(
        job_id=job.job_id,
        status=JobStatus.PENDING,
        message="Job started. Poll /jobs/{job_id} for status.",
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll job status and retrieve model URLs when ready."""
    job = await queue.get_job(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    
    preview_url = None
    final_url = None
    
    if job.preview_path:
        preview_url = f"/outputs/{job_id}/preview.glb"
    if job.final_path:
        final_url = f"/outputs/{job_id}/final.glb"
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        preview_url=preview_url,
        final_url=final_url,
        progress_percent=job.progress_percent,
        message=job.message,
        created_at=str(job.created_at),
        updated_at=str(job.updated_at),
        error=job.error,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check and system status."""
    stats = queue.get_stats()
    return HealthResponse(
        status="healthy",
        models_loaded=coarse_generator._loaded,
        device=DEVICE,
        queued_jobs=stats["queued"],
        active_jobs=stats["active"],
    )
