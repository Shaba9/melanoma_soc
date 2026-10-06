"""FastAPI application entrypoint.

Scaffold only. Endpoints (/predict, /heatmap, /metrics) are implemented in a later
step per API-SPEC.md.
"""
from fastapi import FastAPI

app = FastAPI(title="Melanoma SOC API")


@app.get("/health")
def health():
    return {"status": "ok"}