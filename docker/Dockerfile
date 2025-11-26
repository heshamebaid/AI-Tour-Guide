# Hieroglyph Processing System Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create directories
RUN mkdir -p /app/models /app/data /app/output /app/logs /app/temp_uploads

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy application
WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download models
RUN python -m translation_service.model_downloader sam_vit_b

# Expose API port
EXPOSE 8000

# Start API server
CMD ["python", "-m", "translation_service.api_server"]