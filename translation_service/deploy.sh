#!/bin/bash
# Deployment Script for Hieroglyph Processing System

# Create directories
mkdir -p models data output logs temp_uploads

# Install dependencies
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download models
download-models sam_vit_b  # classifier URL needs to be updated

# Create default config if not exists
if [ ! -f config.yaml ]; then
  cat <<EOT > config.yaml
# Hieroglyph Processing Configuration

# Image processing
IMAGE_SIZE: [512, 512]
CROP_SIZE: [299, 299]
GAUSSIAN_KERNEL: [5, 5]

# SAM parameters
SAM_MODEL_TYPE: "vit_b"
SAM_POINTS_PER_SIDE: 64
SAM_PRED_IOU_THRESH: 0.8
SAM_STABILITY_SCORE_THRESH: 0.85
SAM_CROP_N_LAYERS: 1
SAM_CROP_N_POINTS_DOWNSCALE_FACTOR: 2
SAM_MIN_MASK_REGION_AREA: 70

# Post-processing
MIN_MASK_AREA: 500
CLUSTERING_EPS: 20
CLUSTERING_MIN_SAMPLES: 1

# Story generation
STORY_ENABLED: true
LLM_MODEL: "liquid/lfm-2.5-1.2b-thinking:free"
EOT
fi

echo "Deployment complete. To start API:"
echo "source venv/bin/activate && python -m translation_service.api_server"