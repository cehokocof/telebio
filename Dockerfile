FROM python:3.13-slim AS base

# Prevent Python from writing .pyc and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ─── Dependencies ──────────────────────────────────────
COPY pyproject.toml ./
RUN pip install --no-cache-dir .  2>/dev/null || true

# ─── Application code ─────────────────────────────────
COPY . .

# Install the project itself (editable is not needed in prod)
RUN pip install --no-cache-dir .

# ─── Run ───────────────────────────────────────────────
CMD ["python", "main.py"]
