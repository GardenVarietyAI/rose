from fastapi import Request

from rose_server._inference import InferenceServer

__all__ = ["InferenceServer", "get_inference_server"]


def get_inference_server(request: Request) -> InferenceServer:
    return request.app.state.inference_server
