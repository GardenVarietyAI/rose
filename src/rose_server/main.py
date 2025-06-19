import signal
import sys

import uvicorn

from rose_core.config.service import ServiceConfig


def handle_exit(signum, frame):
    print("\nReceived signal to terminate. Shutting down...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    uvicorn.run(
        "rose_server.app:app",
        host=ServiceConfig.HOST,
        port=ServiceConfig.PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
