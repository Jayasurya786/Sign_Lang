# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    FLASK_APP=app.py

WORKDIR /app

# Install Debian system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for efficient Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and assets
COPY app.py .
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Create data directories for runtime custom dataset recording
RUN mkdir -p data/online_asl data/processed

# Expose web application port
EXPOSE 5000

# Health check to ensure the web server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Start the application
CMD ["python", "app.py"]
