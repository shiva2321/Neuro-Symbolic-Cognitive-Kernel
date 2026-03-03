"""
NSCK Societal Dashboard & API — V26
====================================

FastAPI (or stdlib fallback) REST interface for the Societal Hypervector
Knowledge Representation system.

Endpoints
---------
GET  /societal/status          — society health metrics.
GET  /societal/domain_tree     — hierarchical domain structure.
GET  /societal/communities     — current community (cluster) listing.
POST /societal/register        — register one concept.
POST /societal/query           — query the society with a text concept.
POST /societal/bond            — explicitly bond two concepts.
POST /societal/step            — advance one epoch.
GET  /societal/cluster         — run or retrieve Leiden clustering.
GET  /societal/percolation     — get percolation threshold.
GET  /societal/topo_health     — full topological health report.

Usage::

    server = SocietalDashboard()
    server.run(port=8080)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

_nsck_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _nsck_root not in sys.path:
    sys.path.insert(0, _nsck_root)

logger = logging.getLogger("nsck.societal.dashboard")


class SocietalDashboard:
    """Dashboard / API server for the Societal HV system.

    Parameters
    ----------
    bond_threshold:
        Similarity threshold for auto-bonding.
    max_bonds:
        Maximum bonds per node.
    auto_cluster_interval:
        Epoch interval for auto-clustering (0 = never).
    """

    def __init__(
        self,
        bond_threshold: float = 0.65,
        max_bonds: int = 8,
        auto_cluster_interval: int = 10,
    ) -> None:
        from python.core.societal.society_manager import SocietyManager
        from python.core.societal.societal_context_router import SocietalContextRouter

        self.manager = SocietyManager(
            bond_threshold=bond_threshold,
            max_bonds=max_bonds,
            auto_cluster_interval=auto_cluster_interval,
        )
        self.router = SocietalContextRouter(self.manager)

    # ------------------------------------------------------------------
    # Handler methods (framework-agnostic)
    # ------------------------------------------------------------------

    def handle_status(self) -> Dict[str, Any]:
        """Return society health metrics."""
        return {
            "status": "ok",
            "health": self.manager.topological_health(),
            "summary": self.manager.summary(),
        }

    def handle_domain_tree(self) -> Dict[str, Any]:
        """Return the hierarchical domain structure."""
        return {
            "status": "ok",
            "domain_tree": self.manager.domain_tree(),
        }

    def handle_communities(self) -> Dict[str, Any]:
        """Return current community listing."""
        summary = self.router.get_community_summary()
        return {"status": "ok", "communities": summary}

    def handle_register(self, concept_id: str, domain_path: Optional[List[str]] = None,
                        role: str = "leaf") -> Dict[str, Any]:
        """Register one concept."""
        import python.core.vsa.hypervec_shim as hv_mod
        from python.core.societal.living_hypervector import LivingHyperVector

        hv = hv_mod.HyperVector(seed=abs(hash(concept_id)) % (2 ** 31))
        lhv = LivingHyperVector(
            concept_id=concept_id,
            hv=hv,
            domain_path=domain_path or [],
            role=role,
            birth_epoch=self.manager.epoch,
        )
        self.manager.register(lhv)
        return {
            "status": "ok",
            "concept_id": concept_id,
            "n_concepts": len(self.manager),
        }

    def handle_query(self, concept_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the society with a concept string."""
        import python.core.vsa.hypervec_shim as hv_mod

        # Build a query HV from the concept string seed
        seed = abs(hash(concept_id)) % (2 ** 31)
        query_hv = hv_mod.HyperVector(seed=seed)
        ctx = self.router.route(query_hv, task_tag="dashboard_query")
        return {"status": "ok", "query": concept_id, "context": ctx}

    def handle_bond(self, concept_a: str, concept_b: str,
                    bond_type: str = "similarity",
                    strength: Optional[float] = None) -> Dict[str, Any]:
        """Explicitly bond two concepts."""
        bond = self.manager.form_bond_explicit(
            concept_a, concept_b, bond_type=bond_type, strength=strength
        )
        if bond is None:
            return {"status": "error", "message": "One or both concepts not found."}
        return {
            "status": "ok",
            "bond": {
                "peer_a": concept_a,
                "peer_b": concept_b,
                "strength": bond.strength,
                "bond_type": bond_type,
            },
        }

    def handle_step(self) -> Dict[str, Any]:
        """Advance one epoch."""
        stats = self.manager.step_epoch()
        return {"status": "ok", "epoch_stats": stats}

    def handle_cluster(self, resolution: float = 1.0) -> Dict[str, Any]:
        """Run or retrieve Leiden clustering at given resolution."""
        result = self.manager.leiden_cluster(resolution)
        return {
            "status": "ok",
            "resolution": resolution,
            "n_communities": result.n_communities,
            "modularity": round(result.modularity, 6),
            "communities": {
                str(cid): sorted(members)
                for cid, members in result.communities.items()
            },
        }

    def handle_percolation(self) -> Dict[str, Any]:
        """Return percolation threshold."""
        thr = self.manager.percolation_threshold()
        return {"status": "ok", "percolation_threshold": thr}

    def handle_topo_health(self) -> Dict[str, Any]:
        """Full topological health report."""
        return {"status": "ok", "topo_health": self.manager.topological_health()}

    # ------------------------------------------------------------------
    # FastAPI app factory
    # ------------------------------------------------------------------

    def create_app(self):
        """Create a FastAPI application if FastAPI is installed.

        Returns None if FastAPI is not available.
        """
        try:
            from fastapi import FastAPI  # type: ignore[import]
            from fastapi.responses import JSONResponse  # type: ignore[import]
            from pydantic import BaseModel  # type: ignore[import]
        except ImportError:
            logger.info("FastAPI not installed; use run() for stdlib fallback.")
            return None

        app = FastAPI(
            title="NSCK Societal Dashboard",
            description="Societal Hypervector Knowledge Representation API — NSCK V26",
            version="26.0.0",
        )

        class RegisterRequest(BaseModel):
            concept_id: str
            domain_path: Optional[List[str]] = None
            role: str = "leaf"

        class QueryRequest(BaseModel):
            concept_id: str
            top_k: int = 5

        class BondRequest(BaseModel):
            concept_a: str
            concept_b: str
            bond_type: str = "similarity"
            strength: Optional[float] = None

        class ClusterRequest(BaseModel):
            resolution: float = 1.0

        @app.get("/societal/status")
        def status():
            return JSONResponse(self.handle_status())

        @app.get("/societal/domain_tree")
        def domain_tree():
            return JSONResponse(self.handle_domain_tree())

        @app.get("/societal/communities")
        def communities():
            return JSONResponse(self.handle_communities())

        @app.post("/societal/register")
        def register(req: RegisterRequest):
            return JSONResponse(
                self.handle_register(req.concept_id, req.domain_path, req.role)
            )

        @app.post("/societal/query")
        def query(req: QueryRequest):
            return JSONResponse(self.handle_query(req.concept_id, req.top_k))

        @app.post("/societal/bond")
        def bond(req: BondRequest):
            return JSONResponse(
                self.handle_bond(req.concept_a, req.concept_b, req.bond_type, req.strength)
            )

        @app.post("/societal/step")
        def step():
            return JSONResponse(self.handle_step())

        @app.get("/societal/cluster")
        def cluster(resolution: float = 1.0):
            return JSONResponse(self.handle_cluster(resolution))

        @app.get("/societal/percolation")
        def percolation():
            return JSONResponse(self.handle_percolation())

        @app.get("/societal/topo_health")
        def topo_health():
            return JSONResponse(self.handle_topo_health())

        return app

    # ------------------------------------------------------------------
    # Stdlib fallback HTTP server
    # ------------------------------------------------------------------

    def run(self, host: str = "127.0.0.1", port: int = 8080) -> None:  # pragma: no cover
        """Start the dashboard.  Uses uvicorn if available, else stdlib."""
        app = self.create_app()
        if app is not None:
            try:
                import uvicorn  # type: ignore[import]
                uvicorn.run(app, host=host, port=port)
                return
            except ImportError:
                pass

        # Stdlib fallback
        import http.server
        import urllib.parse

        dashboard = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # suppress default logs
                pass

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path.rstrip("/")
                if path == "/societal/status":
                    body = json.dumps(dashboard.handle_status())
                elif path == "/societal/domain_tree":
                    body = json.dumps(dashboard.handle_domain_tree())
                elif path == "/societal/communities":
                    body = json.dumps(dashboard.handle_communities())
                elif path == "/societal/topo_health":
                    body = json.dumps(dashboard.handle_topo_health())
                elif path == "/societal/percolation":
                    body = json.dumps(dashboard.handle_percolation())
                elif path == "/societal/cluster":
                    qs = urllib.parse.parse_qs(parsed.query)
                    res = float(qs.get("resolution", ["1.0"])[0])
                    body = json.dumps(dashboard.handle_cluster(res))
                else:
                    body = json.dumps({"status": "error", "message": "not found"})
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(body.encode())
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    data = json.loads(raw) if raw else {}
                except Exception:
                    data = {}
                path = self.path.rstrip("/")
                if path == "/societal/register":
                    body = json.dumps(
                        dashboard.handle_register(
                            data.get("concept_id", ""),
                            data.get("domain_path"),
                            data.get("role", "leaf"),
                        )
                    )
                elif path == "/societal/query":
                    body = json.dumps(
                        dashboard.handle_query(
                            data.get("concept_id", ""), data.get("top_k", 5)
                        )
                    )
                elif path == "/societal/bond":
                    body = json.dumps(
                        dashboard.handle_bond(
                            data.get("concept_a", ""),
                            data.get("concept_b", ""),
                            data.get("bond_type", "similarity"),
                            data.get("strength"),
                        )
                    )
                elif path == "/societal/step":
                    body = json.dumps(dashboard.handle_step())
                else:
                    body = json.dumps({"status": "error", "message": "not found"})
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(body.encode())
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())

        server = http.server.HTTPServer((host, port), _Handler)
        logger.info("Societal Dashboard running at http://%s:%d", host, port)
        server.serve_forever()
