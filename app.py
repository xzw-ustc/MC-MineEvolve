"""FastAPI entry point for the MineEvolve server.

Run with::

    uvicorn app:app --port 9000

The default config is read from environment variables (see
``mineevolve.server.api._bootstrap_default_cfg``). For full configurability,
import ``create_app`` directly from ``mineevolve.server`` and pass a config.
"""

from __future__ import annotations

from mineevolve.server import create_app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
    )
