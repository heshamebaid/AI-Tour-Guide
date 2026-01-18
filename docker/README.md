# Docker Setup for AI Tour Guide Platform

This directory contains Docker configuration files for containerizing all services.

## Quick Start

### 1. Create Environment File

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# Required: OPEN_ROUTER_API_KEY
```

### 2. Prepare Data Directory

```bash
# Create data directory for RAG documents
mkdir -p Agentic_RAG/src/controllers/data

# Add your documents (.pdf, .txt, .md files)
# cp your_documents/* Agentic_RAG/src/controllers/data/
```

### 3. Build and Start Services

```bash
# Build all images
docker-compose build

# Start all services in detached mode
docker-compose up -d

# Or start with logs visible
docker-compose up
```

### 4. Access Services

- **Django Web App**: http://localhost:9000/
- **Translation API**: http://localhost:8000/
- **Chatbot API**: http://localhost:8080/
- **Talk To Pharos**: http://localhost:8050/

## Services

### Translation API (Port 8000)
- Translates hieroglyph images
- Endpoint: `/translate`

### Chatbot API (Port 8001)
- RAG-powered chatbot about ancient Egypt
- Endpoint: `/chat`

### Talk To Pharos Service (Port 8050)
- Persona-driven conversations with pharaohs
- Endpoints: `/pharos`, `/converse`, `/health`

### Django Web Server (Port 9000)
- Main web interface
- Integrates all services

## Useful Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f pharos-service

# Rebuild a specific service
docker-compose build pharos-service
docker-compose up -d pharos-service

# Check service status
docker-compose ps

# Execute command in container
docker-compose exec pharos-service bash

# View resource usage
docker stats

# Clean up (remove containers, networks, volumes)
docker-compose down -v
```

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs

# Check if ports are available
docker ps

# Rebuild without cache
docker-compose build --no-cache
```

### RAG not initializing

```bash
# Check pharos service logs
docker-compose logs pharos-service

# Verify data directory is mounted
docker-compose exec pharos-service ls -la Agentic_RAG/src/controllers/data/

# Check environment variables
docker-compose exec pharos-service env | grep OPEN_ROUTER
```

### Django can't connect to services

- Ensure service names match in `docker-compose.yml`
- Use service names (e.g., `pharos-service`) instead of `localhost` in Django settings
- Check that `PHAROS_SERVICE_URL` environment variable is set correctly

## Environment Variables

Required:
- `OPEN_ROUTER_API_KEY` - For RAG services

Optional:
- `PHAROS_SERVICE_URL` - Django uses this to connect to pharos service (default: `http://pharos-service:8050`)
- `DJANGO_SECRET_KEY` - Django secret key
- `DJANGO_DEBUG` - Django debug mode

## Volumes

The following directories are mounted as volumes:
- `./Agentic_RAG/src/controllers/data` - RAG documents
- `./Django` - Django application code
- `./data` - Translation service data
- `./resources` - Translation service resources

## Health Checks

All services have health checks configured:
- Translation API: `/health`
- Chatbot API: `/health`
- Talk To Pharos: `/health`

Health checks run every 30 seconds with appropriate start periods for initialization.

## Production Considerations

For production deployment:
1. Use environment-specific `.env` files
2. Set `DJANGO_DEBUG=False`
3. Configure proper `ALLOWED_HOSTS` in Django settings
4. Use a reverse proxy (nginx) for SSL termination
5. Set up proper logging and monitoring
6. Use Docker secrets for sensitive data
7. Consider using Docker Swarm or Kubernetes for orchestration
