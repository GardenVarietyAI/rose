import uvicorn

from rose_server.config.settings import settings


def main() -> None:
    uvicorn.run(
        "rose_server.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        limit_concurrency=1000,
        timeout_keep_alive=30,
        h11_max_incomplete_event_size=settings.max_file_upload_size,
    )


if __name__ == "__main__":
    main()
