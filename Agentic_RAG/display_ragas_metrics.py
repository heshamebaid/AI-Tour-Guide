"""
Display RAGAS Metrics from evaluation results using Pandas.
"""

import pandas as pd
from pathlib import Path


def display_ragas_metrics(csv_path: str = None):
    """Load and display RAGAS metrics from CSV file."""
    
    # Default path
    if csv_path is None:
        csv_path = Path(__file__).parent / "agentic_ragas_results.csv"
    
    # Load results
    df = pd.read_csv(csv_path)
    
    # Define metric columns
    metric_cols = [
        'faithfulness', 
        'answer_relevancy', 
        'context_precision', 
        'context_recall', 
        'answer_correctness', 
        'answer_similarity'
    ]
    
    # Extract only metrics that exist
    available_metrics = [col for col in metric_cols if col in df.columns]
    
    if not available_metrics:
        print("No RAGAS metrics found in the CSV file.")
        return
    
    # Configure pandas display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Create metrics DataFrame with question summaries
    metrics_df = df[['question'] + available_metrics].copy()
    metrics_df['question'] = metrics_df['question'].str[:55] + '...'
    
    # Display header
    print("\n" + "=" * 120)
    print("📊 RAGAS METRICS PER QUESTION")
    print("=" * 120)
    print(metrics_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("📈 SUMMARY STATISTICS")
    print("=" * 120)
    summary_df = df[available_metrics].agg(['mean', 'min', 'max', 'std']).T
    summary_df.columns = ['Average', 'Min', 'Max', 'Std Dev']
    print(summary_df.to_string())
    
    # Interpretation
    print("\n" + "=" * 120)
    print("📖 METRICS INTERPRETATION")
    print("=" * 120)
    
    interpretations = {
        'faithfulness': 'How well the answer is grounded in the retrieved context (higher = less hallucination)',
        'answer_relevancy': 'How relevant the answer is to the question asked',
        'context_precision': 'Precision of retrieved contexts (are the top results relevant?)',
        'context_recall': 'Recall of retrieved contexts (do they cover the ground truth?)',
        'answer_correctness': 'Factual correctness of the answer compared to ground truth',
        'answer_similarity': 'Semantic similarity between answer and ground truth'
    }
    
    for metric in available_metrics:
        avg = df[metric].mean()
        status = "✅ Good" if avg >= 0.7 else "⚠️ Fair" if avg >= 0.5 else "❌ Needs Improvement"
        print(f"\n  {metric}:")
        print(f"    Score: {avg:.4f} - {status}")
        print(f"    Description: {interpretations.get(metric, 'N/A')}")
    
    print("\n" + "=" * 120)
    print("Score Guide: ≥0.80 Excellent | ≥0.70 Good | ≥0.50 Fair | <0.50 Needs Improvement")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    display_ragas_metrics()
