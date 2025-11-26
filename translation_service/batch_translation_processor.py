#!/usr/bin/env python3
"""
Batch Translation Processor for Hieroglyph Images
Processes all images in Hieroglyph_Images folder and extracts results for academic paper analysis
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "translation_service"

from .hieroglyph_pipeline import HieroglyphPipeline, HieroglyphConfig
from .translation_judge import RAGTranslationJudge, EvaluationCriteria

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/batch_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchTranslationProcessor:
    """Processes multiple hieroglyph images and extracts results for analysis"""
    
    def __init__(self, input_dir: str = "Hieroglyph_Images", output_dir: str = "output/batch_results"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline and evaluation
        self.pipeline = None
        self.evaluator = None
        self.results = []
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'total_symbols_detected': 0,
            'total_processing_time': 0,
            'average_evaluation_score': 0.0
        }
    
    def initialize_components(self):
        """Initialize the translation pipeline and evaluator"""
        try:
            logger.info("Initializing translation pipeline...")
            config = HieroglyphConfig()
            self.pipeline = HieroglyphPipeline(config)
            
            # Setup the pipeline (this initializes SAM, classifier, etc.)
            logger.info("Setting up pipeline components...")
            self.pipeline.setup()
            logger.info("Pipeline initialized successfully")
            
            logger.info("Initializing translation evaluator...")
            self.evaluator = RAGTranslationJudge()
            logger.info("Evaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image and return results"""
        logger.info(f"Processing image: {image_path.name}")
        
        start_time = time.time()
        result = {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'processing_time': 0,
            'success': False,
            'error': None,
            'symbols_detected': 0,
            'classifications': [],
            'translation': '',
            'evaluation': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Process image through pipeline
            pipeline_result = self.pipeline.process_image(
                str(image_path), 
                str(self.output_dir / "symbols" / image_path.stem)
            )
            
            # Extract results
            result['symbols_detected'] = len(pipeline_result.get('classifications', []))
            result['classifications'] = pipeline_result.get('classifications', [])
            result['translation'] = pipeline_result.get('translation', '')
            result['success'] = True
            
            # Evaluate translation if available
            if result['translation'] and self.evaluator:
                try:
                    evaluation = self.evaluator.evaluate_translation(
                        result['classifications'], 
                        result['translation']
                    )
                    result['evaluation'] = {
                        'overall_score': evaluation.overall_score,
                        'criteria_scores': {c.value: s for c, s in evaluation.criteria_scores.items()},
                        'strengths': evaluation.strengths,
                        'weaknesses': evaluation.weaknesses,
                        'suggestions': evaluation.suggestions,
                        'historical_context': evaluation.historical_context,
                        'confidence_analysis': evaluation.confidence_analysis
                    }
                except Exception as e:
                    logger.warning(f"Evaluation failed for {image_path.name}: {e}")
                    result['evaluation'] = None
            
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            result['error'] = str(e)
            result['success'] = False
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def process_all_images(self) -> List[Dict[str, Any]]:
        """Process all images in the input directory"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in self.input_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            raise ValueError(f"No image files found in {self.input_dir}")
        
        logger.info(f"Found {len(image_files)} images to process")
        self.stats['total_images'] = len(image_files)
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing image {i}/{len(image_files)}: {image_path.name}")
            
            try:
                result = self.process_single_image(image_path)
                self.results.append(result)
                
                if result['success']:
                    self.stats['successful_translations'] += 1
                    self.stats['total_symbols_detected'] += result['symbols_detected']
                else:
                    self.stats['failed_translations'] += 1
                
                self.stats['total_processing_time'] += result['processing_time']
                
                # Save individual result
                self.save_individual_result(result)
                
            except Exception as e:
                logger.error(f"Unexpected error processing {image_path.name}: {e}")
                logger.error(traceback.format_exc())
                self.stats['failed_translations'] += 1
        
        # Calculate final statistics
        self.calculate_final_statistics()
        return self.results
    
    def save_individual_result(self, result: Dict[str, Any]):
        """Save individual result to file"""
        result_dir = self.output_dir / "individual_results"
        result_dir.mkdir(exist_ok=True)
        
        result_file = result_dir / f"{Path(result['image_name']).stem}_result.json"
        
        # Convert any non-serializable objects
        serializable_result = self.make_serializable(result)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    
    def make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self.make_serializable(obj.__dict__)
        else:
            return obj
    
    def calculate_final_statistics(self):
        """Calculate final statistics from all results"""
        if not self.results:
            return
        
        # Calculate average evaluation score
        evaluation_scores = [
            r['evaluation']['overall_score'] 
            for r in self.results 
            if r['evaluation'] and 'overall_score' in r['evaluation']
        ]
        
        if evaluation_scores:
            self.stats['average_evaluation_score'] = sum(evaluation_scores) / len(evaluation_scores)
        
        logger.info("Final Statistics:")
        logger.info(f"  Total Images: {self.stats['total_images']}")
        logger.info(f"  Successful: {self.stats['successful_translations']}")
        logger.info(f"  Failed: {self.stats['failed_translations']}")
        logger.info(f"  Total Symbols: {self.stats['total_symbols_detected']}")
        logger.info(f"  Total Time: {self.stats['total_processing_time']:.2f}s")
        logger.info(f"  Average Score: {self.stats['average_evaluation_score']:.2f}/10")
    
    def save_results(self):
        """Save all results and statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        results_file = self.output_dir / f"batch_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': timestamp,
                    'input_directory': str(self.input_dir),
                    'total_images': len(self.results),
                    'processing_time': self.stats['total_processing_time']
                },
                'statistics': self.stats,
                'results': self.make_serializable(self.results)
            }, f, indent=2, ensure_ascii=False)
        
        # Save statistics summary
        stats_file = self.output_dir / f"statistics_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        # Save CSV for analysis
        self.save_csv_results(timestamp)
        
        # Save paper-ready summary
        self.save_paper_summary(timestamp)
        
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"  Complete results: {results_file}")
        logger.info(f"  Statistics: {stats_file}")
    
    def save_csv_results(self, timestamp: str):
        """Save results in CSV format for analysis"""
        # Main results CSV
        csv_data = []
        for result in self.results:
            row = {
                'image_name': result['image_name'],
                'success': result['success'],
                'processing_time': result['processing_time'],
                'symbols_detected': result['symbols_detected'],
                'translation_length': len(result['translation']),
                'overall_score': result['evaluation']['overall_score'] if result['evaluation'] else None,
                'error': result['error']
            }
            
            # Add individual criteria scores
            if result['evaluation'] and 'criteria_scores' in result['evaluation']:
                for criterion, score in result['evaluation']['criteria_scores'].items():
                    row[f'score_{criterion}'] = score
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / f"results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # Symbol-level CSV
        symbol_data = []
        for result in self.results:
            if result['success'] and result['classifications']:
                for i, classification in enumerate(result['classifications']):
                    if 'error' not in classification:
                        symbol_row = {
                            'image_name': result['image_name'],
                            'symbol_index': i,
                            'hieroglyph': classification.get('Hieroglyph', ''),
                            'gardiner_code': classification.get('Gardiner Code', ''),
                            'confidence': classification.get('confidence', 0),
                            'description': classification.get('Description', ''),
                            'details': classification.get('Details', '')
                        }
                        symbol_data.append(symbol_row)
        
        if symbol_data:
            symbol_df = pd.DataFrame(symbol_data)
            symbol_csv_file = self.output_dir / f"symbols_{timestamp}.csv"
            symbol_df.to_csv(symbol_csv_file, index=False)
    
    def save_paper_summary(self, timestamp: str):
        """Save a paper-ready summary of results"""
        summary = {
            'experiment_metadata': {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dataset': str(self.input_dir),
                'total_images': self.stats['total_images'],
                'success_rate': f"{(self.stats['successful_translations'] / self.stats['total_images'] * 100):.1f}%"
            },
            'performance_metrics': {
                'total_processing_time_seconds': round(self.stats['total_processing_time'], 2),
                'average_processing_time_per_image': round(self.stats['total_processing_time'] / self.stats['total_images'], 2),
                'total_symbols_detected': self.stats['total_symbols_detected'],
                'average_symbols_per_image': round(self.stats['total_symbols_detected'] / self.stats['successful_translations'], 1) if self.stats['successful_translations'] > 0 else 0
            },
            'evaluation_metrics': {
                'average_overall_score': round(self.stats['average_evaluation_score'], 2),
                'evaluation_scale': '0-10 (10 being best)',
                'evaluation_criteria': {
                    'historical_accuracy': 'Period-appropriate references and timeline consistency',
                    'cultural_context': 'Ancient Egyptian cultural authenticity',
                    'symbol_meaning_alignment': 'Translation-symbol correspondence',
                    'narrative_coherence': 'Story structure and flow',
                    'egyptological_terminology': 'Proper technical language usage',
                    'confidence_weighting': 'Appropriate use of symbol confidence levels'
                }
            },
            'detailed_results': []
        }
        
        # Add detailed results for each image
        for result in self.results:
            if result['success'] and result['evaluation']:
                detail = {
                    'image': result['image_name'],
                    'symbols_detected': result['symbols_detected'],
                    'overall_score': result['evaluation']['overall_score'],
                    'criteria_scores': result['evaluation']['criteria_scores'],
                    'processing_time': result['processing_time']
                }
                summary['detailed_results'].append(detail)
        
        # Save paper summary
        summary_file = self.output_dir / f"paper_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Also save as markdown for easy reading
        self.save_markdown_summary(summary, timestamp)
    
    def save_markdown_summary(self, summary: Dict, timestamp: str):
        """Save a markdown summary for easy reading"""
        md_content = f"""# Hieroglyph Translation Batch Processing Results

**Date:** {summary['experiment_metadata']['date']}  
**Dataset:** {summary['experiment_metadata']['dataset']}  
**Total Images:** {summary['experiment_metadata']['total_images']}  
**Success Rate:** {summary['experiment_metadata']['success_rate']}

## Performance Metrics

- **Total Processing Time:** {summary['performance_metrics']['total_processing_time_seconds']} seconds
- **Average Time per Image:** {summary['performance_metrics']['average_processing_time_per_image']} seconds
- **Total Symbols Detected:** {summary['performance_metrics']['total_symbols_detected']}
- **Average Symbols per Image:** {summary['performance_metrics']['average_symbols_per_image']}

## Evaluation Results

- **Average Overall Score:** {summary['evaluation_metrics']['average_overall_score']}/10
- **Evaluation Scale:** {summary['evaluation_metrics']['evaluation_scale']}

### Evaluation Criteria

{chr(10).join([f"- **{criterion}:** {description}" for criterion, description in summary['evaluation_metrics']['evaluation_criteria'].items()])}

## Individual Results

| Image | Symbols | Overall Score | Processing Time |
|-------|---------|---------------|-----------------|
"""
        
        for result in summary['detailed_results']:
            md_content += f"| {result['image']} | {result['symbols_detected']} | {result['overall_score']:.2f} | {result['processing_time']:.2f}s |\n"
        
        md_file = self.output_dir / f"paper_summary_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

def main():
    """Main function to run batch processing"""
    try:
        # Initialize processor
        processor = BatchTranslationProcessor()
        
        # Initialize components
        processor.initialize_components()
        
        # Process all images
        logger.info("Starting batch processing...")
        results = processor.process_all_images()
        
        # Save results
        processor.save_results()
        
        logger.info("Batch processing completed successfully!")
        logger.info(f"Processed {len(results)} images")
        logger.info(f"Results saved to: {processor.output_dir}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
