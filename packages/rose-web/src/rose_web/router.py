from typing import Any

from fastapi import APIRouter, Request
from starlette.responses import HTMLResponse, Response, StreamingResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def chat(req: Request) -> Any:
    return req.app.state.templates.TemplateResponse("index.html", {"request": req})


@router.post("/chatkit")
async def chatkit_endpoint(req: Request) -> Any:
    server = req.app.state.chatkit_server
    result = await server.process(await req.body(), {})

    if hasattr(result, "__aiter__"):
        return StreamingResponse(result, media_type="text/event-stream")
    return Response(content=result.json, media_type="application/json")
