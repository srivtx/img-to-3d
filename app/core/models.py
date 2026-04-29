"""Pydantic models for API requests/responses."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING_COARSE = "processing_coarse"
    COARSE_READY = "coarse_ready"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"


class Generate3DResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    preview_url: Optional[str] = Field(None, description="URL to preview GLB model")
    final_url: Optional[str] = Field(None, description="URL to final refined GLB model")
    message: str = Field(..., description="Human-readable status message")


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    preview_url: Optional[str] = None
    final_url: Optional[str] = None
    progress_percent: int = Field(0, ge=0, le=100)
    message: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str
    queued_jobs: int
    active_jobs: int
