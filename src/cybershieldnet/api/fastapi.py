import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from typing import Optional
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Minimal, robust FastAPI app with metrics and version

logger = logging.getLogger("cybershieldnet.api")

def create_app() -> FastAPI:
    app = FastAPI(
        title="CyberShieldNet API",
        description="Multi-Modal Fusion Framework for Predictive Threat Intelligence",
        version="1.0.0"
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

    # Prometheus metrics
    REQ_COUNT = Counter("csn_requests_total", "Total HTTP requests", ["path", "method", "status"])
    REQ_LATENCY = Histogram("csn_request_latency_seconds", "Request latency", ["path"])

    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        path = request.url.path
        with REQ_LATENCY.labels(path=path).time():
            response = await call_next(request)
        REQ_COUNT.labels(path=path, method=request.method, status=response.status_code).inc()
        return response

    @app.get("/")
    async def root():
        return {"message": "CyberShieldNet API", "version": "1.0.0", "status": "operational"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/version")
    async def version():
        return {"name": "CyberShieldNet", "mode": os.getenv("CSN_MODE", "lite")}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Feature-flag complex endpoints to remain compatible in lite mode
    if os.getenv("CSN_ENABLE_COMPLEX", "0") == "1":
        try:
            from .endpoints import router
            app.include_router(router, prefix="/api/v1")
        except Exception:
            pass

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.cybershieldnet.api.fastapi:app", host="0.0.0.0", port=8000, reload=False)