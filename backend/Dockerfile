FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
    "fastapi[standard]" \
    "uvicorn[standard]" \
    python-socketio \
    websockets \
    "torch>=2.2.0" \
    "transformers>=4.37.0" \
    onnxruntime-gpu \
    accelerate \
    bitsandbytes \
    peft \
    huggingface-hub \
    aiohttp \
    redis \
    sqlalchemy \
    alembic \
    python-multipart \
    pydantic-settings \
    python-dotenv \
    librosa \
    soundfile \
    pillow \
    opencv-python \
    firebase-admin

COPY app /app/app
COPY scripts /app/scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
