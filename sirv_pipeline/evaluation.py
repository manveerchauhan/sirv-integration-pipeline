"""
SIRV evaluation module for the SIRV Integration Pipeline.

This module compares expected vs. observed SIRV counts
and provides metrics for evaluating isoform identification tools.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from sirv_pipeline.utils import validate_files

# Set up logger
logger = logging.getLogger(__name__)


def compare_with_flames(
    expected_file: str,
    flames_output: str,
    output_file: str,
    plot_dir: Optional[str] = None
) -> pd.DataFrame:
    """Compare expected vs observed SIRV counts."""
    # Validate inputs
    validate_files(expected_file, flames_output, mode='r')
    validate_files(output_file, mode='w')
    
    logger.info("Comparing expected vs observed SIRV counts...")
    
    # Load expected counts and FLAMES results
    expected = pd.read_csv(expected_file)
    flames = pd.read_csv(flames_output)
    
    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Extract SIRV counts from FLAMES output
    observed = extract_sirv_counts(flames)
    
    # Merge expected and observed
    comparison = pd.merge(
        expected[['barcode', 'sirv_transcript', 'expected_count']],
        observed[['barcode', 'sirv_transcript', 'observed_count']],
        on=['barcode', 'sirv_transcript'],
        how='outer'
    ).fillna(0)
    
    # Calculate metrics
    comparison['detected'] = comparison['observed_count'] > 0
    comparison['detection_rate'] = np.where(
        comparison['expected_count'] > 0,
        comparison['observed_count'] / comparison['expected_count'],
        0
    )
    
    # Save comparison
    comparison.to_csv(output_file, index=False)
    
    # Generate summary
    summary = generate_summary(comparison)
    
    # Print summary
    logger.info(f"SIRV detection summary:")
    logger.info(f"- Expected transcripts: {summary['total_expected']}")
    logger.info(f"- Detected transcripts: {summary['total_detected']}")
    logger.info(f"- Overall detection rate: {summary['detection_rate']:.2%}")
    logger.info(f"- Correlation: {summary['correlation']:.4f}")
    
    # Generate plots if plot directory is provided
    if plot_dir:
        generate_plots(comparison, plot_dir)
        logger.info(f"Plots saved to {plot_dir}")
    
    return comparison


def extract_sirv_counts(flames_df: pd.DataFrame) -> pd.DataFrame:
    """Extract SIRV counts from FLAMES output."""
    # Find required columns by pattern matching
    columns = {
        'transcript': None,
        'barcode': None,
        'count': None
    }
    
    patterns = {
        'transcript': ['transcript'],
        'barcode': ['barcode', 'cell'],
        'count': ['count', 'abundance']
    }
    
    # Find each column type
    for col_type, patterns_list in patterns.items():
        for col in flames_df.columns:
            if any(pattern in col.lower() for pattern in patterns_list):
                columns[col_type] = col
                break
    
    # Check if we found all required columns
    if not all(columns.values()):
        missing = [k for k, v in columns.items() if v is None]
        raise ValueError(f"Could not identify required column(s): {', '.join(missing)}")
    
    # Extract SIRV counts
    transcript_col = columns['transcript']
    barcode_col = columns['barcode']
    count_col = columns['count']
    
    sirv_counts = flames_df[flames_df[transcript_col].str.contains('SIRV', case=False, na=False)].copy()
    
    # Standardize column names
    sirv_counts.rename(columns={
        transcript_col: 'sirv_transcript',
        barcode_col: 'barcode',
        count_col: 'observed_count'
    }, inplace=True)
    
    return sirv_counts[['barcode', 'sirv_transcript', 'observed_count']]


def generate_summary(comparison: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics from comparison data."""
    total_expected = len(comparison[comparison['expected_count'] > 0])
    total_detected = len(comparison[comparison['detected']])
    detection_rate = total_detected / total_expected if total_expected > 0 else 0
    
    # Calculate correlation between expected and observed counts
    expected_counts = comparison['expected_count'].values
    observed_counts = comparison['observed_count'].values
    
    correlation = 0
    if np.sum(expected_counts) > 0 and np.sum(observed_counts) > 0:
        correlation = np.corrcoef(expected_counts, observed_counts)[0, 1]
    
    # Calculate transcript-level metrics
    transcript_metrics = comparison.groupby('sirv_transcript').agg({
        'expected_count': 'sum',
        'observed_count': 'sum',
        'detected': 'mean'
    })
    
    return {
        'total_expected': total_expected,
        'total_detected': total_detected,
        'detection_rate': detection_rate,
        'correlation': correlation,
        'transcript_metrics': transcript_metrics
    }


def generate_plots(comparison: pd.DataFrame, plot_dir: str) -> None:
    """Generate evaluation plots from comparison data."""
    # Create plots directory
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Scatter plot of expected vs observed counts
    plt.figure(figsize=(8, 6))
    plt.scatter(
        comparison['expected_count'], 
        comparison['observed_count'],
        alpha=0.6
    )
    plt.xlabel('Expected SIRV Count')
    plt.ylabel('Observed SIRV Count')
    plt.title('Expected vs Observed SIRV Counts')
    plt.grid(alpha=0.3)
    
    # Add diagonal line
    max_val = max(
        comparison['expected_count'].max(), 
        comparison['observed_count'].max()
    )
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'count_correlation.png'))
    plt.close()
    
    # 2. Detection rate vs expected count
    plt.figure(figsize=(8, 6))
    
    # Add jitter to expected counts for better visualization
    jitter = np.random.normal(0, 0.1, size=len(comparison))
    jittered_expected = comparison['expected_count'] + jitter
    
    plt.scatter(
        jittered_expected, 
        comparison['detected'].astype(int),
        alpha=0.5
    )
    
    plt.xlabel('Expected SIRV Count')
    plt.ylabel('Detected (1=Yes, 0=No)')
    plt.title('SIRV Detection by Expected Count')
    plt.grid(alpha=0.3)
    
    # Add trend line
    bin_means, bin_edges, _ = np.histogram(
        comparison['expected_count'],
        bins=10,
        weights=comparison['detected'].astype(int),
        density=False
    )
    
    bin_counts, _, _ = np.histogram(
        comparison['expected_count'],
        bins=10
    )
    
    bin_means = bin_means / np.maximum(bin_counts, 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    plt.plot(bin_centers, bin_means, 'r-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'detection_rate.png'))
    plt.close()
    
    # 3. Per-transcript detection rates
    transcript_stats = comparison.groupby('sirv_transcript').agg({
        'expected_count': 'sum',
        'observed_count': 'sum',
        'detected': 'mean'
    }).reset_index()
    
    # Sort by detection rate
    transcript_stats = transcript_stats.sort_values('detected', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(len(transcript_stats)), 
        transcript_stats['detected'],
        alpha=0.7
    )
    plt.xlabel('SIRV Transcript')
    plt.ylabel('Detection Rate')
    plt.title('Detection Rate by SIRV Transcript')
    plt.xticks(
        range(len(transcript_stats)), 
        transcript_stats['sirv_transcript'],
        rotation=90
    )
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'transcript_detection.png'))
    plt.close()


def generate_report(
    comparison_file: str,
    output_html: str,
    template_file: Optional[str] = None
) -> str:
    """Generate HTML report of evaluation results."""
    # Validate input files
    validate_files(comparison_file, mode='r')
    validate_files(output_html, mode='w')
    
    # Check if jinja2 is available
    try:
        import jinja2
    except ImportError:
        logger.warning("jinja2 not available, skipping HTML report generation")
        return ""
    
    logger.info("Generating HTML report...")
    
    # Load comparison data and generate summary
    comparison = pd.read_csv(comparison_file)
    summary = generate_summary(comparison)
    
    # Create plots directory and generate plots
    plot_dir = os.path.join(os.path.dirname(output_html), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    generate_plots(comparison, plot_dir)
    
    # Create HTML report template
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SIRV Integration Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            .summary { margin: 20px 0; }
            .plots { display: flex; flex-wrap: wrap; }
            .plot { margin: 10px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1>SIRV Integration Evaluation Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total expected transcripts: {{ summary.total_expected }}</p>
            <p>Total detected transcripts: {{ summary.total_detected }}</p>
            <p>Overall detection rate: {{ "%.2f"|format(summary.detection_rate*100) }}%</p>
            <p>Correlation coefficient: {{ "%.4f"|format(summary.correlation) }}</p>
        </div>
        
        <div class="plots">
            <div class="plot">
                <h2>Expected vs Observed Counts</h2>
                <img src="plots/count_correlation.png" alt="Count Correlation">
            </div>
            <div class="plot">
                <h2>Detection Rate by Expected Count</h2>
                <img src="plots/detection_rate.png" alt="Detection Rate">
            </div>
            <div class="plot">
                <h2>Detection Rate by SIRV Transcript</h2>
                <img src="plots/transcript_detection.png" alt="Transcript Detection">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Use custom template if provided
    if template_file and os.path.exists(template_file):
        with open(template_file, 'r') as f:
            template_str = f.read()
    
    # Render template and write HTML file
    template = jinja2.Template(template_str)
    html_content = template.render(summary=summary, comparison=comparison)
    
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {output_html}")
    return output_html