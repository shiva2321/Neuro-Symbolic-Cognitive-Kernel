"""
NSCK REST API
=============
FastAPI wrapper (optional) with stdlib http.server fallback for NSCK.
"""
from __future__ import annotations

import sys
import os
import json
import logging
from typing import Optional, Any

_nsck_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _nsck_root not in sys.path:
    sys.path.insert(0, _nsck_root)

logger = logging.getLogger("nsck.api")


class NSCKApiServer:
    """REST API wrapper around CognitiveEngine."""

    def __init__(self, config=None):
        from python.core.integration.config import NSCKConfig
        from python.core.reasoning.cognitive_engine import CognitiveEngine

        self._config = config or NSCKConfig()
        self.engine = CognitiveEngine(self._config)

    # ── Handler methods ───────────────────────────────────────────────────────

    def handle_decide(self, state: dict, task_tag: str) -> dict:
        """Run one decide cycle and return structured response."""
        # Ensure task is registered
        if task_tag not in [tb for tb in self.engine.task_brains]:
            self.engine.register_task(task_tag)
        result = self.engine.decide(state, task_tag)
        explanation_text = ""
        active_predicates: list = []
        confidence = getattr(result, "confidence", 0.5)
        action = getattr(result, "chosen_action", "explore")
        if result.explanation is not None:
            explanation_text = str(getattr(result.explanation, "text", result.explanation))
        active_predicates = list(getattr(result, "active_predicates", []))
        return {
            "action": action,
            "confidence": float(confidence),
            "explanation": explanation_text,
            "active_predicates": active_predicates,
        }

    def handle_learn(
        self,
        state: dict,
        action: str,
        reward: float,
        task_tag: str,
        outcome: str = "unknown",
    ) -> dict:
        """Record a learning update."""
        if task_tag not in [tb for tb in self.engine.task_brains]:
            self.engine.register_task(task_tag)
        self.engine.learn(state, action, reward, task_tag)
        return {"status": "ok"}

    def handle_sleep(self, task_tag: Optional[str] = None) -> dict:
        """Run sleep/consolidation cycle."""
        result = self.engine.sleep(task_tag)
        if isinstance(result, dict):
            return result
        return {"status": "sleep_complete"}

    def handle_status(self) -> dict:
        """Return system status."""
        cfg = self._config
        tasks = list(self.engine.task_brains)
        return {
            "status": "running",
            "tasks": tasks,
            "task_count": len(tasks),
            "decisions": self.engine.stats.get("decisions", 0),
            "config": {
                "device": cfg.device,
                "enable_snn": cfg.enable_snn,
                "enable_vsa": cfg.enable_vsa,
            },
        }

    # ── FastAPI app creation ──────────────────────────────────────────────────

    def create_app(self):
        """Create FastAPI app if available, else return None."""
        try:
            from fastapi import FastAPI  # type: ignore
            from pydantic import BaseModel  # type: ignore

            app = FastAPI(title="NSCK API", version="1.0")

            class DecideRequest(BaseModel):
                state: dict
                task_tag: str

            class LearnRequest(BaseModel):
                state: dict
                action: str
                reward: float
                task_tag: str
                outcome: str = "unknown"

            class SleepRequest(BaseModel):
                task_tag: Optional[str] = None

            server = self

            @app.post("/decide")
            def decide(req: DecideRequest):
                return server.handle_decide(req.state, req.task_tag)

            @app.post("/learn")
            def learn(req: LearnRequest):
                return server.handle_learn(
                    req.state, req.action, req.reward, req.task_tag, req.outcome
                )

            @app.post("/sleep")
            def sleep(req: SleepRequest):
                return server.handle_sleep(req.task_tag)

            @app.get("/status")
            def status():
                return server.handle_status()

            return app
        except ImportError:
            return None

    # ── Server runner ─────────────────────────────────────────────────────────

    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """Run the API server.

        Args:
            host: Bind address. Defaults to ``127.0.0.1`` (localhost only).
                  Use ``"0.0.0.0"`` only in trusted network environments, as it
                  exposes the API to all network interfaces.
            port: TCP port to listen on.
        """
        app = self.create_app()
        if app is not None:
            try:
                import uvicorn  # type: ignore
                uvicorn.run(app, host=host, port=port)
                return
            except ImportError:
                pass

        # Fallback: stdlib http.server
        import http.server
        import threading

        server_ref = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                logger.debug(fmt, *args)

            def _send_json(self, data: dict, code: int = 200):
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    return json.loads(self.rfile.read(length))
                return {}

            def do_GET(self):
                if self.path == "/status":
                    self._send_json(server_ref.handle_status())
                else:
                    self._send_json({"error": "Not found"}, 404)

            def do_POST(self):
                data = self._read_json()
                if self.path == "/decide":
                    self._send_json(server_ref.handle_decide(
                        data.get("state", {}), data.get("task_tag", "default")
                    ))
                elif self.path == "/learn":
                    self._send_json(server_ref.handle_learn(
                        data.get("state", {}), data.get("action", ""),
                        float(data.get("reward", 0.0)), data.get("task_tag", "default"),
                        data.get("outcome", "unknown"),
                    ))
                elif self.path == "/sleep":
                    self._send_json(server_ref.handle_sleep(data.get("task_tag")))
                else:
                    self._send_json({"error": "Not found"}, 404)

        httpd = http.server.HTTPServer((host, port), _Handler)
        logger.info(f"NSCK stdlib HTTP server running on {host}:{port}")
        httpd.serve_forever()
