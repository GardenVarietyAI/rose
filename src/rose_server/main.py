import uvicorn

from rose_server.config.settings import settings


def main() -> None:
    uvicorn.run(
        "rose_server.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        limit_max_requests=1000,
        limit_concurrency=1000,
        timeout_keep_alive=30,
        h11_max_incomplete_event_size=100 * 1024 * 1024,  # 100MB for large file uploads
    )


if __name__ == "__main__":
    main()
