"""
api/routes/report.py
────────────────────
UUID-based shareable report storage.
Saves analysis results to disk as JSON. No new DB dependencies.
GET /api/v1/report/{uuid} — retrieve a saved report
POST /api/v1/report       — save a report, get back a UUID + share URL
"""
import uuid
import json
import os
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

# Store reports next to this file in a reports/ directory
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


@router.post("/report")
async def save_report(payload: dict):
    """
    Save an analysis result and return a UUID-based share URL.
    Called from the frontend after analysis completes.
    """
    report_id = str(uuid.uuid4())
    report = {
        "id": report_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data": payload
    }
    path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return {"report_id": report_id, "share_url": f"/report/{report_id}"}


@router.get("/report/{report_id}")
async def get_report(report_id: str):
    """Retrieve a saved report by UUID."""
    # Sanitize: only alphanumeric + hyphens
    safe = "".join(c for c in report_id if c.isalnum() or c == "-")
    path = os.path.join(REPORTS_DIR, f"{safe}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found or expired")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
