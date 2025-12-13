from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
import time


async def homepage(request):
    return JSONResponse(
        {
            "name": "CyberShieldNet API (lite)",
            "status": "ok",
            "timestamp": int(time.time()),
        }
    )


async def health(request):
    return PlainTextResponse("ok")


async def predict(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    return JSONResponse(
        {
            "risk_score": 0.13,
            "details": "API-only mode; machine learning disabled",
            "input_preview": data,
        }
    )


routes = [
    Route("/", endpoint=homepage),
    Route("/health", endpoint=health),
    Route("/predict", endpoint=predict, methods=["POST"]),
]


app = Starlette(routes=routes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tools.starlette_server:app", host="127.0.0.1", port=8000, reload=False)
