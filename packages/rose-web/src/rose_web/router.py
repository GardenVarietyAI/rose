import logging
from typing import Any

from fastapi import APIRouter, Request
from starlette.responses import HTMLResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_class=HTMLResponse)
async def index(req: Request) -> Any:
    return req.app.state.templates.TemplateResponse(
        "index.html",
        {"request": req},
    )


@router.get("/models", response_class=HTMLResponse)
async def list_models(req: Request) -> Any:
    response = await req.app.state.openai.models.list()
    models = response.data
    return req.app.state.templates.TemplateResponse(
        "models.html",
        {"request": req, "models": models},
    )


@router.get("/models/{model_id}", response_class=HTMLResponse)
async def show_model(req: Request, model_id: str) -> Any:
    model = await req.app.state.openai.models.retrieve(model_id)
    return req.app.state.templates.TemplateResponse(
        "models.html",
        {"request": req, "model": model},
    )


@router.delete("/models/{model_id}", response_class=HTMLResponse)
async def delete_model(req: Request, model_id: str) -> Any:
    model = await req.app.state.openai.models.retrieve(model_id)
    return req.app.state.templates.TemplateResponse(
        "models.html",
        {"request": req, "model": model},
    )


@router.get("/fine_tuning", response_class=HTMLResponse)
async def show_fine_tuning_jobs(req: Request) -> Any:
    response = await req.app.state.openai.fine_tuning.jobs.list()
    fine_tuning_jobs = response.data
    return req.app.state.templates.TemplateResponse(
        "fine-tuning-jobs.html",
        {"request": req, "fine_tuning_jobs": fine_tuning_jobs},
    )
