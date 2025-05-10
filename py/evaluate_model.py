
"""
Intent Classification Model Evaluator

This script analyzes the CSV data produced by the Flask app to generate
evaluation metrics and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from datetime import datetime


def load_evaluation_data(csv_path="model_evaluation.csv"):
    """Load the CSV data and do basic preprocessing"""
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found")
        return None
        
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Convert string boolean to actual boolean
	#df['is_ood'] = df['is_ood'].apply(lambda x: x.lower() == 'true')
    df['is_ood'] = df['is_ood'].apply(lambda x: str(x).lower() == 'true')
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} evaluation records")
    return df


def generate_basic_stats(df):
    """Generate basic statistics from the evaluation data"""
    if df is None or len(df) == 0:
        return "No data available for analysis"
        
    stats = {
        "total_queries": len(df),
        "unique_queries": df['input_text'].nunique(),
        "in_distribution_count": (~df['is_ood']).sum(),
        "out_of_distribution_count": df['is_ood'].sum(),
        "ood_percentage": df['is_ood'].mean() * 100,
        "avg_confidence": df['confidence'].mean(),
        "avg_energy_score": df['energy_score'].mean(),
        "top_intents": df['predicted_intent'].value_counts().head(10).to_dict()
    }
    
    # Calculate metrics grouped by detection method
    method_stats = df.groupby('detection_method').agg({
        'is_ood': ['mean', 'count'],
        'confidence': ['mean', 'std'],
        'energy_score': ['mean', 'std']
    })
    
    return stats, method_stats


def plot_distributions(df, output_dir="evaluation_plots"):
    """Create plots for analyzing the model performance"""
    if df is None or len(df) == 0:
        print("No data available for plotting")
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Confidence Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df[~df['is_ood']]['confidence'], bins=20, alpha=0.7, label='In-Distribution')
    plt.hist(df[df['is_ood']]['confidence'], bins=20, alpha=0.7, label='Out-of-Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Score Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{timestamp}_confidence_distribution.png"))
    
    # Plot 2: Energy Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df[~df['is_ood']]['energy_score'], bins=20, alpha=0.7, label='In-Distribution')
    plt.hist(df[df['is_ood']]['energy_score'], bins=20, alpha=0.7, label='Out-of-Distribution')
    plt.xlabel('Energy Score')
    plt.ylabel('Count')
    plt.title('Energy Score Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{timestamp}_energy_distribution.png"))
    
    # Plot 3: Intent Distribution (Top 10)
    intent_counts = df['predicted_intent'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    intent_counts.plot(kind='bar')
    plt.xlabel('Intent')
    plt.ylabel('Count')
    plt.title('Top 10 Predicted Intents')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{timestamp}_intent_distribution.png"))
    
    # Plot 4: OOD Detection Method Comparison
    plt.figure(figsize=(10, 6))
    method_ood = df.groupby('detection_method')['is_ood'].mean() * 100
    method_ood.plot(kind='bar')
    plt.xlabel('Detection Method')
    plt.ylabel('OOD Percentage')
    plt.title('OOD Detection Rate by Method')
    plt.savefig(os.path.join(output_dir, f"{timestamp}_ood_by_method.png"))
    
    print(f"Plots saved to {output_dir} directory")


def analyze_inputs(df):
    """Analyze input texts for patterns"""
    if df is None or len(df) == 0:
        return "No data available for analysis"
        
    # Basic text statistics
    df['text_length'] = df['input_text'].apply(len)
    df['word_count'] = df['input_text'].apply(lambda x: len(x.split()))
    
    text_stats = {
        "avg_text_length": df['text_length'].mean(),
        "avg_word_count": df['word_count'].mean(),
        "max_text_length": df['text_length'].max(),
        "min_text_length": df['text_length'].min()
    }
    
    # Analyze correlation between text length and predictions
    length_vs_ood = df.groupby(pd.cut(df['text_length'], 10))['is_ood'].mean()
    length_vs_confidence = df.groupby(pd.cut(df['text_length'], 10))['confidence'].mean()
    
    print("\nInput Text Analysis:")
    print(f"Average text length: {text_stats['avg_text_length']:.1f} characters")
    print(f"Average word count: {text_stats['avg_word_count']:.1f} words")
    
    return text_stats, length_vs_ood, length_vs_confidence


def suggest_thresholds(df):
    """Analyze the data to suggest optimal thresholds for OOD detection"""
    if df is None or len(df) == 0 or len(df['is_ood'].unique()) < 2:
        return "Insufficient data for threshold analysis - need both OOD and non-OOD examples"
    
    # Simple suggestion based on average values
    suggested_energy = np.mean([
        df[df['is_ood']]['energy_score'].mean(),
        df[~df['is_ood']]['energy_score'].mean()
    ])
    
    suggested_msp = np.mean([
        df[df['is_ood']]['confidence'].mean(),
        df[~df['is_ood']]['confidence'].mean()
    ])
    
    print("\nThreshold Suggestions:")
    print(f"Current data suggests an energy threshold around: {suggested_energy:.4f}")
    print(f"Current data suggests an MSP threshold around: {suggested_msp:.4f}")
    print("Note: These are rough estimates. For proper threshold tuning,")
    print("you should use a dedicated validation set and ROC curve analysis.")
    
    return suggested_energy, suggested_msp


def main():
    parser = argparse.ArgumentParser(description="Analyze intent classification evaluation data")
    parser.add_argument('--csv', default='model_evaluation.csv', help='Path to the evaluation CSV file')
    parser.add_argument('--plots', default='evaluation_plots', help='Directory to save plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    args = parser.parse_args()
    
    print(f"Loading data from {args.csv}...")
    df = load_evaluation_data(args.csv)
    
    if df is not None and len(df) > 0:
        print("\n===== BASIC STATISTICS =====")
        stats, method_stats = generate_basic_stats(df)
        print(f"Total queries: {stats['total_queries']}")
        print(f"In-distribution queries: {stats['in_distribution_count']} ({100-stats['ood_percentage']:.1f}%)")
        print(f"Out-of-distribution queries: {stats['out_of_distribution_count']} ({stats['ood_percentage']:.1f}%)")
        print(f"Average confidence score: {stats['avg_confidence']:.4f}")
        print(f"Average energy score: {stats['avg_energy_score']:.4f}")
        
        print("\nTop predicted intents:")
        for intent, count in list(stats['top_intents'].items())[:5]:
            print(f"  - {intent}: {count}")
            
        print("\n===== DETECTION METHOD COMPARISON =====")
        print(method_stats)
        
        # Analyze input texts
        analyze_inputs(df)
        
        # Suggest threshold values
        suggest_thresholds(df)
        
        # Generate plots if not disabled
        if not args.no_plots:
            plot_distributions(df, args.plots)
            
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()