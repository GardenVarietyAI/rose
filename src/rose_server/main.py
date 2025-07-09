import uvicorn

from rose_server.config.settings import settings


def main():
    uvicorn.run(
        "rose_server.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
