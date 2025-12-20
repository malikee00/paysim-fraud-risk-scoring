# =========================
# Base image
# =========================
FROM python:3.11-slim

# =========================
# Workdir
# =========================
WORKDIR /app

# =========================
# System dependencies
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

# =========================
# Python dependencies
# =========================
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# Copy project code
# =========================
COPY . /app

# =========================
# Environment
# =========================
ENV PYTHONPATH=/app
ENV PORT=8000

# =========================
# Expose port
# =========================
EXPOSE 8000

# =========================
# Start API
# =========================
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
