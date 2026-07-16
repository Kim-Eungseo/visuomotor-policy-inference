"""Pydantic request/response schemas for the inference server."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """One control-step observation.

    Images are base64-encoded PNG/JPEG bytes (any resolution — the policy resizes
    internally). State is the raw joint vector.
    """

    front_rgb: str = Field(..., description="base64-encoded PNG/JPEG, scene/zivid camera (RGB)")
    wrist_rgb: str = Field(..., description="base64-encoded PNG/JPEG, wrist camera (RGB)")
    state: list[float] = Field(..., description="robot state vector (arm 6 + hand 20 = 26)")
    gripper_sensor: list[float] | None = Field(
        default=None, description="gripper tactile sensor vector (30); required by *_full / DINOv3 models")
    wrist_ft_sensor: list[float] | None = Field(
        default=None, description="wrist force/torque vector (6); required by *_full / DINOv3 models")
    eef: list[float] | None = Field(
        default=None, description="end-effector pose vector (6); required by some diffusion checkpoints")
    grip: list[float] | None = Field(
        default=None, description="gripper scalar (1); required by some diffusion checkpoints")


class PredictResponse(BaseModel):
    action: list[float] = Field(..., description="target joints (arm 6 + hand 20 = 26)")


class LoadRequest(BaseModel):
    """Switch the served policy at runtime (used by the GUI model picker)."""

    model_id: str = Field(..., description="HF model id or local checkpoint path (diffusion)")
    policy: str = Field(default="diffusion", description="'diffusion' | 'knn'")
    step_mode: bool = Field(
        default=False, description="re-infer every /predict (n_action_steps=1, RTC off) for step-wise closed loop")


class LoadResponse(BaseModel):
    status: str
    info: dict


class InfoResponse(BaseModel):
    model_id: str
    device: str
    cameras: list[str]
    state_dim: int
    action_dim: int
    scheduler: str
    num_inference_steps: int | None
    ruckig: dict | None = Field(default=None, description="jerk-limited smoothing config, or null if disabled")
    rtc: dict | None = Field(default=None, description="real-time chunking (async RTC) config, or null if disabled")
    knn: dict | None = Field(default=None, description="kNN (VINN) retrieval config, or null if not a kNN policy")
    step_mode: bool = Field(default=False, description="true if re-inferring every predict (step-wise closed loop)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ResetResponse(BaseModel):
    status: str
