import uvicorn

from rose_core.config.service import HOST, PORT


def main():
    uvicorn.run(
        "rose_server.app:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
