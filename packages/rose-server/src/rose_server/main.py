import argparse
import logging
from typing import Optional, Sequence


def serve(
    host: str,
    port: int,
    log_level: str,
) -> None:
    import uvicorn

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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="rose-server")
    parser.add_argument("--host", "-H", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", "-p", type=int, default=8004, help="Server port")
    parser.add_argument("--log-level", "-l", default="INFO", help="Logging level")

    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
