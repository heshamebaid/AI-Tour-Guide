"""
Batch processing script for multiple hieroglyph images
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "translation_service"

from .hieroglyph_pipeline import HieroglyphPipeline, HieroglyphConfig, HieroglyphLogger


def setup_batch_logging(log_file: str = "batch_processing.log"):
    """Setup logging for batch processing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_image_files(input_dir: str) -> List[str]:
    """Get all image files from input directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


def process_batch(input_dir: str, output_dir: str, config_file: str = None):
    """Process multiple images in batch."""
    logger = setup_batch_logging()
    
    try:
        # Initialize pipeline
        config = HieroglyphConfig()
        if config_file and os.path.exists(config_file):
            # Load custom config if provided
            logger.info(f"Loading config from {config_file}")
        
        pipeline = HieroglyphPipeline(config)
        
        # Setup pipeline
        pipeline.setup(
            sam_checkpoint="models/sam_vit_b.pth",
            model_path="models/InceptionV3_model.h5",
            gardiner_path="data/Alan_Gardiners_List_of_Hieroglyphic_Signs.xlsx"
        )
        
        # Get image files
        image_files = get_image_files(input_dir)
        logger.info(f"Found {len(image_files)} images to process")
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return
        
        # Process each image
        batch_results = []
        success_count = 0
        error_count = 0
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_path}")
            
            try:
                # Create output subdirectory for this image
                image_name = Path(image_path).stem
                image_output_dir = os.path.join(output_dir, f"image_{i:03d}_{image_name}")
                
                # Process image
                result = pipeline.process_image(image_path, image_output_dir)
                result['batch_index'] = i
                result['original_path'] = image_path
                
                batch_results.append(result)
                success_count += 1
                
                logger.info(f"Successfully processed {image_path} - found {result['num_symbols_found']} symbols")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                batch_results.append({
                    'batch_index': i,
                    'original_path': image_path,
                    'error': str(e),
                    'status': 'failed'
                })
                error_count += 1
        
        # Save batch summary
        batch_summary = {
            'total_images': len(image_files),
            'successful': success_count,
            'failed': error_count,
            'results': batch_results
        }
        
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing complete: {success_count} successful, {error_count} failed")
        logger.info(f"Summary saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise


def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description="Batch process hieroglyph images")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory for output results")
    parser.add_argument("--config", help="Path to config file", default=None)
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process batch
    process_batch(args.input_dir, args.output_dir, args.config)


if __name__ == "__main__":
    main()