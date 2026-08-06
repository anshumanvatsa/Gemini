"""
Entrypoint to run the PreViral API locally.
Run: python run.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
