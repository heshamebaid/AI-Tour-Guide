#!/usr/bin/env python3
"""
Paper Results Extractor
Extracts and formats results from batch processing for academic paper writing
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def load_latest_results(results_dir: str = "output/batch_results") -> Dict[str, Any]:
    """Load the most recent batch processing results"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find the most recent batch results file
    batch_files = list(results_path.glob("batch_results_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No batch results found in {results_dir}")
    
    latest_file = max(batch_files, key=lambda f: f.stat().st_mtime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_performance_table(results: Dict[str, Any]) -> str:
    """Generate a performance metrics table for the paper"""
    stats = results['statistics']
    
    table = """
## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Images Processed | {total_images} |
| Successful Translations | {successful} |
| Failed Translations | {failed} |
| Success Rate | {success_rate:.1f}% |
| Total Symbols Detected | {total_symbols} |
| Average Symbols per Image | {avg_symbols:.1f} |
| Total Processing Time | {total_time:.2f} seconds |
| Average Time per Image | {avg_time:.2f} seconds |
| Average Evaluation Score | {avg_score:.2f}/10 |
""".format(
        total_images=stats['total_images'],
        successful=stats['successful_translations'],
        failed=stats['failed_translations'],
        success_rate=(stats['successful_translations'] / stats['total_images'] * 100),
        total_symbols=stats['total_symbols_detected'],
        avg_symbols=stats['total_symbols_detected'] / stats['successful_translations'] if stats['successful_translations'] > 0 else 0,
        total_time=stats['total_processing_time'],
        avg_time=stats['total_processing_time'] / stats['total_images'],
        avg_score=stats['average_evaluation_score']
    )
    
    return table

def generate_evaluation_analysis(results: Dict[str, Any]) -> str:
    """Generate evaluation criteria analysis"""
    results_data = results['results']
    
    # Calculate average scores for each criterion
    criteria_scores = {}
    total_evaluations = 0
    
    for result in results_data:
        if result.get('evaluation') and 'criteria_scores' in result['evaluation']:
            total_evaluations += 1
            for criterion, score in result['evaluation']['criteria_scores'].items():
                if criterion not in criteria_scores:
                    criteria_scores[criterion] = []
                criteria_scores[criterion].append(score)
    
    if not criteria_scores:
        return "No evaluation data available."
    
    analysis = """
## Evaluation Criteria Analysis

The translation quality was evaluated across six criteria using a 0-10 scale:

| Criterion | Average Score | Description |
|-----------|---------------|-------------|
"""
    
    criterion_descriptions = {
        'historical_accuracy': 'Period-appropriate references and timeline consistency',
        'cultural_context': 'Ancient Egyptian cultural authenticity',
        'symbol_meaning_alignment': 'Translation-symbol correspondence',
        'narrative_coherence': 'Story structure and flow',
        'egyptological_terminology': 'Proper technical language usage',
        'confidence_weighting': 'Appropriate use of symbol confidence levels'
    }
    
    for criterion, scores in criteria_scores.items():
        avg_score = sum(scores) / len(scores)
        description = criterion_descriptions.get(criterion, 'N/A')
        analysis += f"| {criterion.replace('_', ' ').title()} | {avg_score:.2f} | {description} |\n"
    
    analysis += f"\n**Total Evaluations:** {total_evaluations}\n"
    
    return analysis

def generate_individual_results_table(results: Dict[str, Any]) -> str:
    """Generate a table of individual image results"""
    results_data = results['results']
    
    table = """
## Individual Image Results

| Image | Symbols | Overall Score | Processing Time | Status |
|-------|---------|---------------|-----------------|--------|
"""
    
    for result in sorted(results_data, key=lambda x: x['image_name']):
        status = "Success" if result['success'] else "Failed"
        overall_score = result['evaluation']['overall_score'] if result.get('evaluation') else "N/A"
        processing_time = f"{result['processing_time']:.2f}s"
        
        table += f"| {result['image_name']} | {result['symbols_detected']} | {overall_score} | {processing_time} | {status} |\n"
    
    return table

def generate_symbol_analysis(results: Dict[str, Any]) -> str:
    """Generate symbol detection analysis"""
    results_data = results['results']
    
    # Analyze symbol detection
    symbol_counts = [r['symbols_detected'] for r in results_data if r['success']]
    
    if not symbol_counts:
        return "No successful symbol detections to analyze."
    
    analysis = f"""
## Symbol Detection Analysis

- **Total Symbols Detected:** {sum(symbol_counts)}
- **Average Symbols per Image:** {sum(symbol_counts) / len(symbol_counts):.1f}
- **Minimum Symbols:** {min(symbol_counts)}
- **Maximum Symbols:** {max(symbol_counts)}
- **Images with Symbols:** {len(symbol_counts)}/{len(results_data)}
"""
    
    return analysis

def generate_latex_tables(results: Dict[str, Any]) -> str:
    """Generate LaTeX tables for academic papers"""
    stats = results['statistics']
    
    latex = """
% Performance Metrics Table
\\begin{table}[h]
\\centering
\\caption{System Performance Metrics}
\\begin{tabular}{|l|c|}
\\hline
\\textbf{Metric} & \\textbf{Value} \\\\
\\hline
Total Images Processed & """ + str(stats['total_images']) + """ \\\\
Successful Translations & """ + str(stats['successful_translations']) + """ \\\\
Success Rate & """ + f"{(stats['successful_translations'] / stats['total_images'] * 100):.1f}%" + """ \\\\
Total Symbols Detected & """ + str(stats['total_symbols_detected']) + """ \\\\
Average Processing Time & """ + f"{stats['total_processing_time'] / stats['total_images']:.2f}s" + """ \\\\
Average Evaluation Score & """ + f"{stats['average_evaluation_score']:.2f}/10" + """ \\\\
\\hline
\\end{tabular}
\\end{table}

% Individual Results Table (first 10 images)
\\begin{table}[h]
\\centering
\\caption{Sample Translation Results}
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Image} & \\textbf{Symbols} & \\textbf{Score} & \\textbf{Time (s)} \\\\
\\hline
"""
    
    # Add first 10 successful results
    successful_results = [r for r in results['results'] if r['success'] and r.get('evaluation')][:10]
    
    for result in successful_results:
        image_name = result['image_name'].replace('_', '\\_')  # Escape underscores for LaTeX
        symbols = result['symbols_detected']
        score = f"{result['evaluation']['overall_score']:.2f}"
        time_val = f"{result['processing_time']:.2f}"
        latex += f"{image_name} & {symbols} & {score} & {time_val} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    
    return latex

def main():
    """Main function to extract and format results"""
    print("=" * 60)
    print("PAPER RESULTS EXTRACTOR")
    print("=" * 60)
    
    try:
        # Load latest results
        print("Loading latest batch processing results...")
        results = load_latest_results()
        print(f"Loaded results from {results['metadata']['timestamp']}")
        
        # Generate different output formats
        output_dir = Path("output/paper_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate markdown report
        markdown_content = f"""# Hieroglyph Translation System Evaluation Results

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Dataset:** {results['metadata']['input_directory']}  
**Processing Date:** {results['metadata']['timestamp']}

{generate_performance_table(results)}

{generate_evaluation_analysis(results)}

{generate_individual_results_table(results)}

{generate_symbol_analysis(results)}

## Conclusion

The hieroglyph translation system processed {results['statistics']['total_images']} images with a success rate of {(results['statistics']['successful_translations'] / results['statistics']['total_images'] * 100):.1f}%. The average evaluation score of {results['statistics']['average_evaluation_score']:.2f}/10 indicates {'good' if results['statistics']['average_evaluation_score'] >= 7 else 'moderate' if results['statistics']['average_evaluation_score'] >= 5 else 'poor'} translation quality across the evaluation criteria.
"""
        
        # Save markdown report
        md_file = output_dir / f"paper_results_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Save LaTeX tables
        latex_file = output_dir / f"latex_tables_{timestamp}.tex"
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(generate_latex_tables(results))
        
        # Save CSV for further analysis
        csv_file = output_dir / f"detailed_results_{timestamp}.csv"
        df = pd.DataFrame(results['results'])
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults extracted and saved to: {output_dir}")
        print(f"Files generated:")
        print(f"  - {md_file.name} (markdown report)")
        print(f"  - {latex_file.name} (LaTeX tables)")
        print(f"  - {csv_file.name} (detailed CSV data)")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("SUMMARY FOR PAPER")
        print("=" * 60)
        print(f"Total Images: {results['statistics']['total_images']}")
        print(f"Success Rate: {(results['statistics']['successful_translations'] / results['statistics']['total_images'] * 100):.1f}%")
        print(f"Average Score: {results['statistics']['average_evaluation_score']:.2f}/10")
        print(f"Total Symbols: {results['statistics']['total_symbols_detected']}")
        print(f"Processing Time: {results['statistics']['total_processing_time']:.2f}s")
        
    except Exception as e:
        print(f"Error extracting results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

