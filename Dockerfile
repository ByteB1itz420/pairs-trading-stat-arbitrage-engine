FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[backend]"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn pairs_trading.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
