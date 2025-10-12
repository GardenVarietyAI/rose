import uvicorn

from rose_web.settings import settings


def main() -> None:
    uvicorn.run(
        "rose_web.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        limit_concurrency=1000,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
