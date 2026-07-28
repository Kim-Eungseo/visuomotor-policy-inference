"""FastAPI server for serving the Hyundai Uiwang FlowMatch Diffusion Policy.

Endpoints:
    GET  /health   -> liveness + model-loaded flag
    GET  /info      -> model metadata (cameras, dims, scheduler)
    POST /reset     -> clear the action queue (start a new episode)
    POST /predict   -> one observation -> one 26-d action
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from app.policy_runner import PolicyRunner, decode_image
from app.schemas import (
    HealthResponse,
    InfoResponse,
    LoadRequest,
    LoadResponse,
    PredictRequest,
    PredictResponse,
    ResetResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vpi")

STATE: dict[str, PolicyRunner] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    if os.environ.get("VPI_POLICY", "diffusion").lower() == "knn":
        from app.knn_runner import KNNPolicyRunner
        logger.info("Loading kNN (VINN) policy ...")
        STATE["runner"] = KNNPolicyRunner()
    else:
        logger.info("Loading diffusion policy ...")
        STATE["runner"] = PolicyRunner()
    logger.info("Policy ready: %s", STATE["runner"].info())
    yield
    STATE.clear()


app = FastAPI(title="Visuomotor Policy Inference", version="1.0.0", lifespan=lifespan)


def get_runner() -> PolicyRunner:
    runner = STATE.get("runner")
    if runner is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return runner


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded="runner" in STATE)


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    return InfoResponse(**get_runner().info())


@app.post("/reset", response_model=ResetResponse)
def reset() -> ResetResponse:
    get_runner().reset()
    return ResetResponse(status="reset")


@app.post("/load", response_model=LoadResponse)
def load(req: LoadRequest) -> LoadResponse:
    """Swap the served policy at runtime (GUI model picker). diffusion loads `model_id`;
    knn re-inits from the VPI_KNN_* env."""
    try:
        if req.policy.lower() == "knn":
            from app.knn_runner import KNNPolicyRunner
            logger.info("Loading kNN (VINN) policy ...")
            STATE["runner"] = KNNPolicyRunner()
        else:
            logger.info("Loading diffusion policy from %s (step_mode=%s) ...", req.model_id, req.step_mode)
            STATE["runner"] = PolicyRunner(model_id=req.model_id, step_mode=req.step_mode)
    except Exception as e:  # noqa: BLE001
        logger.exception("model load failed")
        raise HTTPException(status_code=500, detail=f"load failed: {e}") from e
    info = STATE["runner"].info()
    logger.info("Policy switched: %s", info)
    return LoadResponse(status="loaded", info=info)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    runner = get_runner()
    try:
        front = decode_image(req.front_rgb)
        wrist = decode_image(req.wrist_rgb)
        state = np.asarray(req.state, dtype=np.float32)
        extra_inputs = {}
        if req.eef is not None:
            extra_inputs["observation.eef"] = req.eef
        if req.grip is not None:
            extra_inputs["observation.grip"] = req.grip
        action = runner.predict(front, wrist, state,
                                gripper_sensor=req.gripper_sensor, wrist_ft_sensor=req.wrist_ft_sensor,
                                extra_inputs=extra_inputs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail=f"inference error: {e}") from e
    retrieved = getattr(runner, "last_retrieval", None)   # kNN runner only; None for diffusion
    return PredictResponse(action=action.tolist(), retrieved=retrieved)
