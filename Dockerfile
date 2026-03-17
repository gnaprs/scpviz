FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages commonly needed by scientific Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Install project + production web server.
RUN python -m pip install --upgrade pip && \
    python -m pip install -e . && \
    python -m pip install gunicorn

EXPOSE 8050

# Single worker keeps in-memory per-session state consistent.
CMD ["sh", "-c", "gunicorn dash_app.app:server --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-8050}"]
