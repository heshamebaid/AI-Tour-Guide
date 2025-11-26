FROM python:3.10-slim

WORKDIR /app

# Install system deps for sentence-transformers / FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal files first to leverage Docker cache
COPY talk_to_pharos_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire repo so the Agentic_RAG package is available
COPY . /app

ENV PHAROS_HOST=0.0.0.0
ENV PHAROS_PORT=8050

EXPOSE 8050

CMD ["uvicorn", "talk_to_pharos_service.app:app", "--host", "0.0.0.0", "--port", "8050"]

