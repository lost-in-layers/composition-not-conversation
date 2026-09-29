"""Local entry point: binds strictly to 127.0.0.1.

Usage:  python -m hr_backend.main [--port 8791]
"""
from __future__ import annotations

import argparse

import uvicorn

from .api import create_app

DEFAULT_PORT = 8791


def main() -> None:
    parser = argparse.ArgumentParser(description="MuLAn human-review local server (localhost only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
