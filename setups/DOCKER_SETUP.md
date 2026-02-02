# Docker Setup Guide

Complete Docker setup for the AI Tour Guide Platform with all services. **Run from the repository root** (where this file and `docker-compose.yml` live).

## Venv and Git

- Virtual environments (`venv/`, `.venv/`, etc.) are **ignored by Git** and **excluded from the Docker build context** (`.dockerignore`), so you can push without venv and builds stay fast.
- If venv was ever committed, run once: `ensure_no_venv_in_git.bat` (Windows) or `./ensure_no_venv_in_git.sh` (Linux/Mac), then commit the change.

## Prerequisites

- Docker installed and running
- Docker Compose installed
- At least 4GB RAM available
- OpenRouter API key

## Requirements (per service)

All Python requirements are installed **inside** the images during `docker-compose build`. You do not need to run `pip install` on your host.

| Service            | Requirements files used in Dockerfile |
|--------------------|----------------------------------------|
| **translation-api** | `requirements.txt` (root) |
| **chatbot-api**     | `requirements.txt`, `Chatbot/requirements.txt`, `Agentic_RAG/requirements.txt` |
| **pharos-service**  | `requirements.txt`, `talk_to_pharos_service/requirements.txt`, `Agentic_RAG/requirements.txt` |
| **django-web**      | `requirements.txt` (root) + `django`, `djangorestframework` |

All of these files exist in the repo. Build order uses root `requirements.txt` first, then service-specific files where applicable.

## Quick Start

### 1. Set Up Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# OPEN_ROUTER_API_KEY=your_actual_api_key_here
```

### 2. Prepare Data Directory

```bash
# Create data directory for RAG documents
mkdir -p Agentic_RAG/src/controllers/data

# Add your documents (.pdf, .txt, .md files)
# Example:
# cp your_documents/*.pdf Agentic_RAG/src/controllers/data/
```

### 3. Build and Start All Services

```bash
# Build all Docker images
docker-compose build

# Start all services in detached mode
docker-compose up -d

# Or start with logs visible (recommended for first run)
docker-compose up
```

### 4. Verify Services Are Running

```bash
# Check service status
docker-compose ps

# Check logs (run from project root D:\AI-Tour-Guide)
docker-compose logs -f
# Or last 100 lines for specific services:
docker-compose logs --tail=100 chatbot-api pharos-service django-web place-details-service
# Follow one service:
docker-compose logs -f chatbot-api

# Test health endpoints
curl http://localhost:8000/health  # Translation API
curl http://localhost:8080/health  # Chatbot API
curl http://localhost:8050/health  # Talk To Pharos
curl http://localhost:9000/       # Django
```

## Access Points

Once all services are running:

- **Main Web Interface**: http://localhost:9000/
- **Talk To Pharos**: http://localhost:9000/talk-to-pharos/
- **Translation API**: http://localhost:8000/
- **Chatbot API**: http://localhost:8080/
- **Talk To Pharos API**: http://localhost:8050/

## Service Details

### Translation API (Port 8000)
- Translates hieroglyph images
- Health: `GET /health`
- Translate: `POST /translate`

### Chatbot API (Port 8080)
- RAG-powered chatbot about ancient Egypt
- Health: `GET /health`
- Chat: `POST /chat`

### Talk To Pharos Service (Port 8050)
- Persona-driven conversations with pharaohs
- Health: `GET /health`
- List Personas: `GET /pharos`
- Converse: `POST /converse`

### Django Web Server (Port 9000)
- Main web interface integrating all services
- Access: http://localhost:9000/

## Common Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f pharos-service

# Rebuild a specific service
docker-compose build pharos-service
docker-compose up -d pharos-service

# Rebuild all services
docker-compose build --no-cache
docker-compose up -d

# Check resource usage
docker stats

# Execute command in container
docker-compose exec pharos-service bash
docker-compose exec django-web python manage.py shell

# Stop and remove everything (including volumes)
docker-compose down -v
```

## Troubleshooting

### Services Won't Start

```bash
# Check logs for errors
docker-compose logs

# Check if ports are already in use
# Windows:
netstat -ano | findstr :8000
netstat -ano | findstr :8080
netstat -ano | findstr :8050
netstat -ano | findstr :9000

# Linux/Mac:
lsof -i :8000
lsof -i :8080
lsof -i :8050
lsof -i :9000

# Rebuild without cache
docker-compose build --no-cache
```

### RAG Not Initializing

```bash
# Check pharos service logs
docker-compose logs pharos-service

# Verify data directory exists and has files
docker-compose exec pharos-service ls -la Agentic_RAG/src/controllers/data/

# Check environment variables
docker-compose exec pharos-service env | grep OPEN_ROUTER

# Check if documents are loading
docker-compose logs pharos-service | grep "Loading documents"
```

### Django Can't Connect to Services

- Ensure service names match in `docker-compose.yml`
- Django uses service names (e.g., `pharos-service`) not `localhost`
- Check `PHAROS_SERVICE_URL` environment variable in Django container

### Health Checks Failing

```bash
# Check individual service health
docker-compose exec translation-api curl http://localhost:8000/health
docker-compose exec chatbot-api curl http://localhost:8080/health
docker-compose exec pharos-service curl http://localhost:8050/health

# Increase health check start period if services need more time
# Edit docker-compose.yml healthcheck.start_period
```

### Out of Memory

```bash
# Check Docker resource usage
docker stats

# Increase Docker memory limit in Docker Desktop settings
# Recommended: At least 4GB RAM
```

## Environment Variables

### Required
- `OPEN_ROUTER_API_KEY` - OpenRouter API key for RAG services

### Optional
- `PHAROS_SERVICE_URL` - Django uses this to connect to pharos service (default: `http://pharos-service:8050`)
- `DJANGO_SECRET_KEY` - Django secret key
- `DJANGO_DEBUG` - Django debug mode (default: True)

## Volumes

The following directories are mounted as volumes for persistence:

- `./Agentic_RAG/src/controllers/data` - RAG documents (persists across restarts)
- `./Django` - Django application code (hot-reload enabled)
- `./data` - Translation service data
- `./resources` - Translation service resources

## Production Deployment

For production:

1. **Set proper environment variables**:
   ```bash
   DJANGO_DEBUG=False
   DJANGO_SECRET_KEY=<strong-secret-key>
   ```

2. **Update ALLOWED_HOSTS** in Django settings:
   ```python
   ALLOWED_HOSTS = ['your-domain.com', 'www.your-domain.com']
   ```

3. **Use a reverse proxy** (nginx) for SSL termination

4. **Set up proper logging**:
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "3"
   ```

5. **Use Docker secrets** for sensitive data

6. **Consider Docker Swarm or Kubernetes** for orchestration

## Clean Up

```bash
# Stop and remove containers
docker-compose down

# Stop and remove containers, networks, and volumes
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Complete cleanup (containers, networks, volumes, images)
docker-compose down -v --rmi all
```

## Support

For issues:
1. Check service logs: `docker-compose logs <service-name>`
2. Verify environment variables are set correctly
3. Ensure data directory has documents
4. Check Docker resource limits
