# PreViral — Production Dockerfile for Google Cloud Run
# "Built with Gemini, deployed on Google Cloud"
FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (production — CPU-only torch)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code + model files + hashtag DB
COPY api/ api/
COPY engines/ engines/
COPY counterfactual/ counterfactual/
COPY frontend/ frontend/
COPY hashtag_db/ hashtag_db/
COPY data/ data/
COPY models/__init__.py models/__init__.py
COPY models/train_lstm.py models/train_lstm.py
COPY models/saved/previral_lgbm_v5.joblib models/saved/previral_lgbm_v5.joblib
COPY models/saved/feature_columns_v5.joblib models/saved/feature_columns_v5.joblib
COPY models/saved/trajectory_lstm_best.pt models/saved/trajectory_lstm_best.pt
COPY models/saved/trajectory_scaler.joblib models/saved/trajectory_scaler.joblib
COPY models/saved/trajectory_target_max.joblib models/saved/trajectory_target_max.joblib

# Expose the port Cloud Run will use
ENV PORT=8080
EXPOSE 8080

# Start the server
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
