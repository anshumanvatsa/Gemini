"""
FastAPI Main App — PreViral
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from api.routes import analyze, hashtags, media


app = FastAPI(
    title="PreViral API",
    description="Pre-publication social media performance predictor",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models load lazily on first request (already cached from previous run)

# Mount static frontend — frontend/ lives at previral root, one level above api/
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Include routes
app.include_router(analyze.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(hashtags.router, prefix="/api/v1", tags=["Hashtags"])
app.include_router(media.router, prefix="/api/v1", tags=["Media"])

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/api/v1/status")
async def status():
    return {"status": "online", "version": "1.0.0", "service": "PreViral"}
