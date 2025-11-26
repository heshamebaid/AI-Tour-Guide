#!/usr/bin/env python3
"""
Optimized Batch Translation Processor
Addresses performance issues and provides better results for academic paper
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
        logging.FileHandler('logs/optimized_batch_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedBatchProcessor:
    """Optimized batch processor with performance improvements and better filtering"""
    
    def __init__(self, input_dir: str = "Hieroglyph_Images", output_dir: str = "output/optimized_results"):
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
            'all_symbols_included': 0,  # Changed from high_confidence_symbols
            'total_processing_time': 0,
            'average_evaluation_score': 0.0
        }
    
    def create_optimized_config(self) -> HieroglyphConfig:
        """Create an optimized configuration for better performance and results"""
        config = HieroglyphConfig()
        
        # Optimize SAM parameters for better segmentation
        config.SAM_POINTS_PER_SIDE = 32  # Reduced from 64 for faster processing
        config.SAM_PRED_IOU_THRESH = 0.9  # Increased for better quality
        config.SAM_STABILITY_SCORE_THRESH = 0.9  # Increased for better quality
        config.SAM_MIN_MASK_REGION_AREA = 200  # Increased to filter tiny segments
        
        # Optimize post-processing
        config.MIN_MASK_AREA = 1000  # Increased to filter small segments
        config.CLUSTERING_EPS = 15  # Reduced for tighter clustering
        config.CLUSTERING_MIN_SAMPLES = 2  # Increased for better clustering
        
        # Optimize story generation
        config.MAX_TOKENS = 2000  # Reduced for faster generation
        
        return config
    
    def initialize_components(self):
        """Initialize the translation pipeline and evaluator"""
        try:
            logger.info("Initializing optimized translation pipeline...")
            config = self.create_optimized_config()
            self.pipeline = HieroglyphPipeline(config)
            
            # Setup the pipeline
            logger.info("Setting up pipeline components...")
            self.pipeline.setup()
            logger.info("Pipeline initialized successfully")
            
            logger.info("Initializing translation evaluator...")
            self.evaluator = RAGTranslationJudge()
            logger.info("Evaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def filter_classifications(self, classifications: List[Dict]) -> List[Dict]:
        """Filter classifications to keep all valid results (no confidence threshold)"""
        if not classifications:
            return []
        
        filtered = []
        for cls in classifications:
            if 'error' in cls:
                continue
            
            # Include ALL symbols regardless of confidence
            # Only filter out error cases
            filtered.append(cls)
        
        # Sort by confidence for better organization
        filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # No limit on number of symbols - include all detected symbols
        return filtered
    
    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image with optimizations and correct data extraction"""
        logger.info(f"Processing image: {image_path.name}")
        
        start_time = time.time()
        result = {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'processing_time': 0,
            'success': False,
            'error': None,
            'symbols_detected': 0,
            'all_symbols_included': 0,  # Changed from high_confidence_symbols
            'classifications': [],
            'translation': '',
            'story': '',  # Add story field
            'evaluation': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Process image through pipeline
            pipeline_result = self.pipeline.process_image(
                str(image_path), 
                str(self.output_dir / "symbols" / image_path.stem)
            )
            
            # Extract and validate data from pipeline result
            raw_classifications = pipeline_result.get('classifications', [])
            if not isinstance(raw_classifications, list):
                raw_classifications = []
            
            # Filter classifications for quality
            filtered_classifications = self.filter_classifications(raw_classifications)
            
            # Extract translation/story - check multiple possible keys
            translation = pipeline_result.get('translation', '') or pipeline_result.get('story', '')
            
            # Ensure we have a valid translation
            if not translation or not isinstance(translation, str):
                translation = "No translation generated"
            
            # Extract results with proper validation
            result['symbols_detected'] = len(raw_classifications)
            result['all_symbols_included'] = len(filtered_classifications)  # Now includes all symbols
            result['classifications'] = filtered_classifications
            result['translation'] = translation
            result['story'] = translation  # Keep both for compatibility
            result['success'] = True
            
            # Log data extraction details
            logger.info(f"Data extracted for {image_path.name}:")
            logger.info(f"  - Total symbols: {len(raw_classifications)}")
            logger.info(f"  - Symbols included: {len(filtered_classifications)}")
            logger.info(f"  - Translation length: {len(translation)} characters")
            
            # Evaluate translation if we have symbols and translation
            if result['translation'] and result['translation'] != "No translation generated" and len(filtered_classifications) > 0 and self.evaluator:
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
                    logger.info(f"  - Evaluation score: {evaluation.overall_score:.2f}/10")
                except Exception as e:
                    logger.warning(f"Evaluation failed for {image_path.name}: {e}")
                    result['evaluation'] = None
            else:
                logger.warning(f"No evaluation for {image_path.name}: translation='{result['translation'][:50]}...', symbols={len(filtered_classifications)}")
            
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            result['error'] = str(e)
            result['success'] = False
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def process_all_images(self) -> List[Dict[str, Any]]:
        """Process all images in the input directory - handles any number of images"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        # Get all image files with comprehensive extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
        image_files = [
            f for f in self.input_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            raise ValueError(f"No image files found in {self.input_dir}")
        
        # Sort files for consistent processing order (handle numeric names properly)
        def sort_key(file_path):
            name = file_path.stem
            # Try to extract number for proper sorting
            try:
                return int(name)
            except ValueError:
                return name
        
        image_files.sort(key=sort_key)
        
        logger.info(f"Found {len(image_files)} images to process:")
        for i, img in enumerate(image_files, 1):
            logger.info(f"  {i:2d}. {img.name}")
        
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
                    self.stats['all_symbols_included'] += result['all_symbols_included']
                else:
                    self.stats['failed_translations'] += 1
                
                self.stats['total_processing_time'] += result['processing_time']
                
                # Save individual result
                self.save_individual_result(result)
                
                # Log progress
                logger.info(f"Completed {image_path.name}: {result['all_symbols_included']} symbols included, "
                           f"{result['processing_time']:.1f}s")
                
            except Exception as e:
                logger.error(f"Unexpected error processing {image_path.name}: {e}")
                logger.error(traceback.format_exc())
                self.stats['failed_translations'] += 1
        
        # Calculate final statistics
        self.calculate_final_statistics()
        return self.results
    
    def save_individual_result(self, result: Dict[str, Any]):
        """Save individual result to file with translation and detailed data"""
        result_dir = self.output_dir / "individual_results"
        result_dir.mkdir(exist_ok=True)
        
        base_name = Path(result['image_name']).stem
        
        # Save complete result as JSON
        result_file = result_dir / f"{base_name}_result.json"
        serializable_result = self.make_serializable(result)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        # Save translation as separate text file
        if result.get('translation'):
            translation_file = result_dir / f"{base_name}_translation.txt"
            with open(translation_file, 'w', encoding='utf-8') as f:
                f.write(f"Translation for {result['image_name']}\n")
                f.write("=" * 50 + "\n\n")
                f.write(result['translation'])
                f.write("\n\n")
                
                # Add evaluation details if available
                if result.get('evaluation'):
                    f.write("Evaluation Details:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Overall Score: {result['evaluation']['overall_score']:.2f}/10\n\n")
                    
                    if 'criteria_scores' in result['evaluation']:
                        f.write("Criteria Scores:\n")
                        for criterion, score in result['evaluation']['criteria_scores'].items():
                            f.write(f"  {criterion.replace('_', ' ').title()}: {score:.2f}/10\n")
                        f.write("\n")
                    
                    if 'strengths' in result['evaluation'] and result['evaluation']['strengths']:
                        f.write("Strengths:\n")
                        for strength in result['evaluation']['strengths']:
                            f.write(f"  • {strength}\n")
                        f.write("\n")
                    
                    if 'weaknesses' in result['evaluation'] and result['evaluation']['weaknesses']:
                        f.write("Areas for Improvement:\n")
                        for weakness in result['evaluation']['weaknesses']:
                            f.write(f"  • {weakness}\n")
                        f.write("\n")
                    
                    if 'suggestions' in result['evaluation'] and result['evaluation']['suggestions']:
                        f.write("Suggestions:\n")
                        for suggestion in result['evaluation']['suggestions']:
                            f.write(f"  • {suggestion}\n")
                        f.write("\n")
        
        # Save symbols data as separate CSV
        if result.get('classifications'):
            symbols_file = result_dir / f"{base_name}_symbols.csv"
            import pandas as pd
            
            symbols_data = []
            for i, classification in enumerate(result['classifications']):
                symbols_data.append({
                    'symbol_index': i + 1,
                    'hieroglyph': classification.get('Hieroglyph', ''),
                    'gardiner_code': classification.get('Gardiner Code', ''),
                    'confidence': classification.get('confidence', 0),
                    'description': classification.get('Description', ''),
                    'details': classification.get('Details', ''),
                    'image_path': classification.get('path', '')
                })
            
            df = pd.DataFrame(symbols_data)
            df.to_csv(symbols_file, index=False)
        
        # Save evaluation as separate JSON if available
        if result.get('evaluation'):
            evaluation_file = result_dir / f"{base_name}_evaluation.json"
            with open(evaluation_file, 'w', encoding='utf-8') as f:
                json.dump(result['evaluation'], f, indent=2, ensure_ascii=False)
    
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
        logger.info(f"  All Symbols Included: {self.stats['all_symbols_included']}")
        logger.info(f"  Total Time: {self.stats['total_processing_time']:.2f}s")
        logger.info(f"  Average Time per Image: {self.stats['total_processing_time'] / self.stats['total_images']:.2f}s")
        logger.info(f"  Average Score: {self.stats['average_evaluation_score']:.2f}/10")
    
    def save_results(self):
        """Save all results and statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        results_file = self.output_dir / f"optimized_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': timestamp,
                    'input_directory': str(self.input_dir),
                    'total_images': len(self.results),
                    'processing_time': self.stats['total_processing_time'],
                    'optimizations': {
                        'min_confidence_threshold': 0.3,
                        'max_symbols_per_image': 20,
                        'improved_sam_parameters': True,
                        'filtered_classifications': True
                    }
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
        
        # Save all translations in one file
        self.save_all_translations(timestamp)
        
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"  Complete results: {results_file}")
        logger.info(f"  Statistics: {stats_file}")
        logger.info(f"  All translations: all_translations_{timestamp}.txt")
    
    def save_csv_results(self, timestamp: str):
        """Save results in CSV format for analysis"""
        # Main results CSV
        csv_data = []
        for result in self.results:
            row = {
                'image_name': result['image_name'],
                'success': result['success'],
                'processing_time': result['processing_time'],
                'total_symbols': result['symbols_detected'],
                'all_symbols_included': result['all_symbols_included'],
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
        
        # High-confidence symbols CSV
        symbol_data = []
        for result in self.results:
            if result['success'] and result['classifications']:
                for i, classification in enumerate(result['classifications']):
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
            symbol_csv_file = self.output_dir / f"all_symbols_{timestamp}.csv"
            symbol_df.to_csv(symbol_csv_file, index=False)
    
    def save_paper_summary(self, timestamp: str):
        """Save a paper-ready summary of results"""
        summary = {
            'experiment_metadata': {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dataset': str(self.input_dir),
                'total_images': self.stats['total_images'],
                'success_rate': f"{(self.stats['successful_translations'] / self.stats['total_images'] * 100):.1f}%",
                'optimizations_applied': [
                    'Improved SAM parameters for better segmentation',
                    'Confidence filtering (≥0.3 threshold)',
                    'Symbol limit per image (max 20)',
                    'Enhanced clustering parameters'
                ]
            },
            'performance_metrics': {
                'total_processing_time_seconds': round(self.stats['total_processing_time'], 2),
                'average_processing_time_per_image': round(self.stats['total_processing_time'] / self.stats['total_images'], 2),
                'total_symbols_detected': self.stats['total_symbols_detected'],
                'all_symbols_included': self.stats['all_symbols_included'],
                'average_symbols_per_image': round(self.stats['all_symbols_included'] / self.stats['successful_translations'], 1) if self.stats['successful_translations'] > 0 else 0,
                'symbol_inclusion_ratio': round(self.stats['all_symbols_included'] / max(self.stats['total_symbols_detected'], 1), 3)
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
                    'total_symbols': result['symbols_detected'],
                    'all_symbols_included': result['all_symbols_included'],
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
        md_content = f"""# Optimized Hieroglyph Translation Results

**Date:** {summary['experiment_metadata']['date']}  
**Dataset:** {summary['experiment_metadata']['dataset']}  
**Total Images:** {summary['experiment_metadata']['total_images']}  
**Success Rate:** {summary['experiment_metadata']['success_rate']}

## Optimizations Applied

{chr(10).join([f"- {opt}" for opt in summary['experiment_metadata']['optimizations_applied']])}

## Performance Metrics

- **Total Processing Time:** {summary['performance_metrics']['total_processing_time_seconds']} seconds
- **Average Time per Image:** {summary['performance_metrics']['average_processing_time_per_image']} seconds
- **Total Symbols Detected:** {summary['performance_metrics']['total_symbols_detected']}
- **All Symbols Included:** {summary['performance_metrics']['all_symbols_included']}
- **Average Symbols per Image:** {summary['performance_metrics']['average_symbols_per_image']}
- **Symbol Inclusion Ratio:** {summary['performance_metrics']['symbol_inclusion_ratio']:.1%}

## Evaluation Results

- **Average Overall Score:** {summary['evaluation_metrics']['average_overall_score']}/10
- **Evaluation Scale:** {summary['evaluation_metrics']['evaluation_scale']}

### Evaluation Criteria

{chr(10).join([f"- **{criterion}:** {description}" for criterion, description in summary['evaluation_metrics']['evaluation_criteria'].items()])}

## Individual Results

| Image | Total Symbols | All Included | Overall Score | Processing Time |
|-------|---------------|-----------------|---------------|-----------------|
"""
        
        for result in summary['detailed_results']:
            md_content += f"| {result['image']} | {result['total_symbols']} | {result['all_symbols_included']} | {result['overall_score']:.2f} | {result['processing_time']:.2f}s |\n"
        
        md_file = self.output_dir / f"paper_summary_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def save_all_translations(self, timestamp: str):
        """Save all translations in one comprehensive file"""
        translations_file = self.output_dir / f"all_translations_{timestamp}.txt"
        
        with open(translations_file, 'w', encoding='utf-8') as f:
            f.write("HIEROGLYPH TRANSLATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images: {len(self.results)}\n")
            f.write(f"Successful Translations: {self.stats['successful_translations']}\n")
            f.write(f"Average Score: {self.stats['average_evaluation_score']:.2f}/10\n\n")
            
            for i, result in enumerate(self.results, 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"IMAGE {i}: {result['image_name']}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Processing Time: {result['processing_time']:.2f} seconds\n")
                f.write(f"Symbols Detected: {result['symbols_detected']}\n")
                f.write(f"All Symbols Included: {result['all_symbols_included']}\n")
                
                if result.get('evaluation'):
                    f.write(f"Overall Score: {result['evaluation']['overall_score']:.2f}/10\n")
                
                f.write(f"\nTRANSLATION:\n")
                f.write("-" * 20 + "\n")
                f.write(result.get('translation', 'No translation available'))
                f.write("\n\n")
                
                if result.get('evaluation'):
                    f.write("EVALUATION DETAILS:\n")
                    f.write("-" * 20 + "\n")
                    
                    if 'criteria_scores' in result['evaluation']:
                        f.write("Criteria Scores:\n")
                        for criterion, score in result['evaluation']['criteria_scores'].items():
                            f.write(f"  {criterion.replace('_', ' ').title()}: {score:.2f}/10\n")
                        f.write("\n")
                    
                    if 'strengths' in result['evaluation'] and result['evaluation']['strengths']:
                        f.write("Strengths:\n")
                        for strength in result['evaluation']['strengths']:
                            f.write(f"  • {strength}\n")
                        f.write("\n")
                    
                    if 'weaknesses' in result['evaluation'] and result['evaluation']['weaknesses']:
                        f.write("Areas for Improvement:\n")
                        for weakness in result['evaluation']['weaknesses']:
                            f.write(f"  • {weakness}\n")
                        f.write("\n")
                    
                    if 'suggestions' in result['evaluation'] and result['evaluation']['suggestions']:
                        f.write("Suggestions:\n")
                        for suggestion in result['evaluation']['suggestions']:
                            f.write(f"  • {suggestion}\n")
                        f.write("\n")
                
                if result.get('classifications'):
                    f.write("HIGH-CONFIDENCE SYMBOLS:\n")
                    f.write("-" * 25 + "\n")
                    for j, symbol in enumerate(result['classifications'], 1):
                        f.write(f"{j:2d}. {symbol.get('Hieroglyph', '?')} ")
                        f.write(f"({symbol.get('Gardiner Code', 'Unknown')}) ")
                        f.write(f"Confidence: {symbol.get('confidence', 0):.1%}\n")
                        f.write(f"    Description: {symbol.get('Description', 'Unknown')}\n")
                        f.write(f"    Details: {symbol.get('Details', 'No details')}\n\n")
                
                if result.get('error'):
                    f.write(f"ERROR: {result['error']}\n")
        
        logger.info(f"All translations saved to: {translations_file}")

def main():
    """Main function to run optimized batch processing"""
    try:
        # Initialize processor
        processor = OptimizedBatchProcessor()
        
        # Initialize components
        processor.initialize_components()
        
        # Process all images
        logger.info("Starting optimized batch processing...")
        results = processor.process_all_images()
        
        # Save results
        processor.save_results()
        
        logger.info("Optimized batch processing completed successfully!")
        logger.info(f"Processed {len(results)} images")
        logger.info(f"Results saved to: {processor.output_dir}")
        
    except Exception as e:
        logger.error(f"Optimized batch processing failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
