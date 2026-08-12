"""
FastAPI Main App — PreViral x Gemini
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from api.routes import analyze, hashtags, media, report

app = FastAPI(
    title="PreViral API",
    description="Pre-publication social media performance predictor powered by Gemini",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Routes
app.include_router(analyze.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(hashtags.router, prefix="/api/v1", tags=["Hashtags"])
app.include_router(media.router,    prefix="/api/v1", tags=["Media"])
app.include_router(report.router,   prefix="/api/v1", tags=["Reports"])

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# Serve report pages via the same SPA — JS handles routing
@app.get("/report/{report_id}", include_in_schema=False)
async def serve_report_page(report_id: str):
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/api/v1/status")
async def status():
    return {"status": "online", "version": "2.0.0", "service": "PreViral",
            "gemini": "gemini-flash-latest"}
