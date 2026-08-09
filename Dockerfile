# --- Stage 1: build the frontend ---
FROM node:24-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: python runtime serving API + built frontend ---
# Pinned to 3.12 for scientific/geo wheel availability (trimesh, manifold3d, etc.).
FROM python:3.12-slim AS runtime
WORKDIR /app

# CJK font for the underside 出典 stamp (app/core/stamp.py); DejaVu is the
# ASCII fallback for CJK-less dev boxes, tiny enough to keep for parity.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-noto-cjk fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY --from=frontend /frontend/dist ./static

ENV STATIC_DIR=/app/static \
    DATA_DIR=/app/data \
    PORT=8000
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Stage 3: the same runtime, plus the test dependencies ---
# The suite runs against the image it ships in — same pinned wheels, same CJK
# font the 出典 stamp needs — and never touches the network. docker-compose's
# `test` service mounts backend/ over this, so editing a test needs no rebuild.
FROM runtime AS test
COPY backend/requirements-dev.txt backend/pytest.ini ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY backend/tests ./tests
ENV DATA_DIR=/tmp/3dfp-test-data
CMD ["python", "-m", "pytest"]
