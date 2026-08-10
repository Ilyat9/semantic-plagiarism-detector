FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PIP_NO_CACHE_DIR=1

# Системные зависимости для WeasyPrint (pango/cairo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN pip install uv && uv sync --no-dev --no-install-project

COPY . .

# Предзагрузка ML-моделей при сборке образа (bi-encoder + cross-encoder)
RUN .venv/bin/python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'); \
CrossEncoder('cross-encoder/stsb-roberta-base', device='cpu')"

EXPOSE 8000 8501

CMD [".venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
