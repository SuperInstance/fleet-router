#!/usr/bin/env python3
"""fleet_router.cli — Start the fleet router server."""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Fleet Router API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "fleet_router.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
