#!/bin/bash

# Docker Startup Script for AI Tour Guide Platform

echo "🏺 AI Tour Guide Platform - Docker Setup"
echo "========================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "📝 Creating .env from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file"
        echo "⚠️  Please edit .env and add your OPEN_ROUTER_API_KEY"
        echo "   Then run this script again."
        exit 1
    else
        echo "❌ .env.example not found!"
        exit 1
    fi
fi

# Check if OPEN_ROUTER_API_KEY is set
if ! grep -q "OPEN_ROUTER_API_KEY=.*[^your_openrouter_api_key_here]" .env; then
    echo "⚠️  OPEN_ROUTER_API_KEY not set in .env file!"
    echo "   Please edit .env and add your API key"
    exit 1
fi

# Create data directory if it doesn't exist
echo "📁 Creating data directory..."
mkdir -p Agentic_RAG/src/controllers/data

# Check if data directory is empty
if [ -z "$(ls -A Agentic_RAG/src/controllers/data)" ]; then
    echo "⚠️  Data directory is empty!"
    echo "   Add .pdf, .txt, or .md files to Agentic_RAG/src/controllers/data/"
    echo "   (You can continue, but RAG won't have documents to search)"
fi

# Build and start services
echo ""
echo "🔨 Building Docker images..."
docker-compose build

echo ""
echo "🚀 Starting all services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 10

echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "✅ Services started!"
echo ""
echo "📍 Access Points:"
echo "   • Django Web App:      http://localhost:9000/"
echo "   • Talk To Pharos:      http://localhost:9000/talk-to-pharos/"
echo "   • Translation API:     http://localhost:8000/"
echo "   • Chatbot API:         http://localhost:8080/"
echo "   • Talk To Pharos API:  http://localhost:8050/"
echo ""
echo "📋 Useful Commands:"
echo "   • View logs:           docker-compose logs -f"
echo "   • Stop services:       docker-compose down"
echo "   • Restart services:    docker-compose restart"
echo ""
