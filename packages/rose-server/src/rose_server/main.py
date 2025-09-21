import logging

import typer
import uvicorn


def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Server host"),
    port: int = typer.Option(8004, "--port", "-p", help="Server port"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level", case_sensitive=False),
) -> None:
    log_level = log_level.upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvicorn.run(
        "rose_server.app:app",
        host=host,
        port=port,
        reload=False,
        limit_concurrency=1000,
        timeout_keep_alive=30,
        h11_max_incomplete_event_size=100 * 1024 * 1024,
        log_level=log_level.lower(),
    )


def main() -> None:
    typer.run(serve)


if __name__ == "__main__":
    main()
