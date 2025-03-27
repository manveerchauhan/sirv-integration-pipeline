"""
SIRV evaluation module for the SIRV Integration Pipeline.

This module compares expected vs. observed SIRV counts
and provides metrics and visualizations for evaluating
isoform identification tools like FLAMES.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union

# Set up logger
logger = logging.getLogger(__name__)


def compare_with_flames(
    expected_file: str,
    flames_output: str,
    output_file: str,
    plot_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare expected vs observed SIRV counts.
    
    Args:
        expected_file: Path to expected counts CSV
        flames_output: Path to FLAMES output CSV
        output_file: Path to output comparison CSV
        plot_dir: Directory to save plots (optional)
        
    Returns:
        pd.DataFrame: Comparison DataFrame
        
    Raises:
        FileNotFoundError: If input files do not exist
    """
    # Validate inputs
    for input_file, description in [
        (expected_file, "Expected counts"), 
        (flames_output, "FLAMES output")
    ]:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"{description} file not found: {input_file}")
    
    logger.info("Comparing expected vs observed SIRV counts...")
    
    # Load expected counts
    try:
        expected = pd.read_csv(expected_file)
    except Exception as e:
        logger.error(f"Error loading expected counts: {e}")
        raise
    
    # Load FLAMES results
    try:
        flames = pd.read_csv(flames_output)
    except Exception as e:
        logger.error(f"Error loading FLAMES results: {e}")
        raise
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Extract SIRV counts from FLAMES output
    observed = _extract_sirv_counts(flames)
    
    # Merge expected and observed
    comparison = pd.merge(
        expected[['barcode', 'sirv_transcript', 'expected_count']],
        observed[['barcode', 'sirv_transcript', 'observed_count']],
        on=['barcode', 'sirv_transcript'],
        how='outer'
    ).fillna(0)
    
    # Calculate metrics
    comparison['detected'] = comparison['observed_count'] > 0
    comparison['detection_rate'] = comparison.apply(
        lambda row: row['observed_count'] / row['expected_count'] 
        if row['expected_count'] > 0 else 0,
        axis=1
    )
    
    # Save comparison
    comparison.to_csv(output_file, index=False)
    
    # Generate summary
    summary = _generate_summary(comparison)
    
    # Print summary
    logger.info(f"SIRV detection summary:")
    logger.info(f"- Expected transcripts: {summary['total_expected']}")
    logger.info(f"- Detected transcripts: {summary['total_detected']}")
    logger.info(f"- Overall detection rate: {summary['detection_rate']:.2%}")
    logger.info(f"- Correlation: {summary['correlation']:.4f}")
    
    # Generate plots if plot directory is provided
    if plot_dir is not None:
        _generate_plots(comparison, plot_dir)
    
    return comparison


def _extract_sirv_counts(flames_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract SIRV counts from FLAMES output.
    
    Args:
        flames_df: FLAMES output DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with SIRV counts
    """
    # Check column names and adapt accordingly
    if 'transcript_id' in flames_df.columns:
        transcript_col = 'transcript_id'
    elif 'transcript' in flames_df.columns:
        transcript_col = 'transcript'
    else:
        # Try to find a column containing transcript information
        for col in flames_df.columns:
            if 'transcript' in col.lower():
                transcript_col = col
                break
        else:
            raise ValueError("Could not find transcript column in FLAMES output")
    
    if 'cell_barcode' in flames_df.columns:
        barcode_col = 'cell_barcode'
    elif 'barcode' in flames_df.columns:
        barcode_col = 'barcode'
    elif 'cell' in flames_df.columns:
        barcode_col = 'cell'
    else:
        # Try to find a column containing cell or barcode information
        for col in flames_df.columns:
            if 'cell' in col.lower() or 'barcode' in col.lower():
                barcode_col = col
                break
        else:
            raise ValueError("Could not find barcode column in FLAMES output")
    
    if 'count' in flames_df.columns:
        count_col = 'count'
    elif 'umi_count' in flames_df.columns:
        count_col = 'umi_count'
    elif 'counts' in flames_df.columns:
        count_col = 'counts'
    else:
        # Try to find a column containing count information
        for col in flames_df.columns:
            if 'count' in col.lower() or 'abundance' in col.lower():
                count_col = col
                break
        else:
            raise ValueError("Could not find count column in FLAMES output")
    
    # Extract SIRV counts
    sirv_counts = flames_df[flames_df[transcript_col].str.contains('SIRV', case=False, na=False)].copy()
    
    # Standardize column names
    sirv_counts.rename(columns={
        transcript_col: 'sirv_transcript',
        barcode_col: 'barcode',
        count_col: 'observed_count'
    }, inplace=True)
    
    return sirv_counts[['barcode', 'sirv_transcript', 'observed_count']]


def _generate_summary(comparison: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics from comparison.
    
    Args:
        comparison: Comparison DataFrame
        
    Returns:
        Dict[str, Any]: Dictionary of summary statistics
    """
    total_expected = len(comparison[comparison['expected_count'] > 0])
    total_detected = len(comparison[comparison['detected']])
    detection_rate = total_detected / total_expected if total_expected > 0 else 0
    
    # Calculate correlation between expected and observed counts
    expected_counts = comparison['expected_count'].values
    observed_counts = comparison['observed_count'].values
    
    if np.sum(expected_counts) > 0 and np.sum(observed_counts) > 0:
        correlation = np.corrcoef(expected_counts, observed_counts)[0, 1]
    else:
        correlation = 0
    
    # Calculate transcript-level metrics
    transcript_metrics = comparison.groupby('sirv_transcript').agg({
        'expected_count': 'sum',
        'observed_count': 'sum',
        'detected': 'mean'
    })
    
    # Calculate cell-level metrics
    cell_metrics = comparison.groupby('barcode').agg({
        'expected_count': 'sum',
        'observed_count': 'sum',
        'detected': 'mean'
    })
    
    return {
        'total_expected': total_expected,
        'total_detected': total_detected,
        'detection_rate': detection_rate,
        'correlation': correlation,
        'transcript_metrics': transcript_metrics,
        'cell_metrics': cell_metrics
    }


def _generate_plots(comparison: pd.DataFrame, plot_dir: str) -> None:
    """
    Generate evaluation plots from comparison data.
    
    Args:
        comparison: Comparison DataFrame
        plot_dir: Directory to save plots
    """
    # 1. Detection rate by expected count
    plt.figure(figsize=(10, 6))
    
    # Add jitter to expected counts for better visualization
    jitter = np.random.normal(0, 0.1, size=len(comparison))
    jittered_expected = comparison['expected_count'] + jitter
    
    plt.scatter(
        jittered_expected, 
        comparison['detected'].astype(int),
        alpha=0.5
    )
    
    # Add trend line
    from scipy.stats import binned_statistic
    bins = 10
    bin_means, bin_edges, _ = binned_statistic(
        comparison['expected_count'],
        comparison['detected'].astype(int),
        statistic='mean',
        bins=bins
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, bin_means, 'r-', linewidth=2)
    
    plt.xlabel('Expected Count')
    plt.ylabel('Detection Rate')
    plt.title('SIRV Detection Rate by Expected Count')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'detection_rate_by_count.png'), dpi=300)
    plt.close()
    
    # 2. Expected vs. Observed counts
    plt.figure(figsize=(10, 6))
    plt.scatter(
        comparison['expected_count'],
        comparison['observed_count'],
        alpha=0.5
    )
    
    # Add diagonal line
    max_count = max(comparison['expected_count'].max(), comparison['observed_count'].max())
    plt.plot([0, max_count], [0, max_count], 'k--', alpha=0.5)
    
    plt.xlabel('Expected Count')
    plt.ylabel('Observed Count')
    plt.title('Expected vs. Observed SIRV Counts')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'expected_vs_observed.png'), dpi=300)
    plt.close()
    
    # 3. Detection rate by transcript
    transcript_metrics = comparison.groupby('sirv_transcript').agg({
        'expected_count': 'sum',
        'observed_count': 'sum',
        'detected': 'mean'
    }).reset_index()
    
    # Sort by detection rate
    transcript_metrics = transcript_metrics.sort_values('detected', ascending=False)
    
    # Plot top 20 transcripts by detection rate
    top_n = min(20, len(transcript_metrics))
    plt.figure(figsize=(12, 8))
    plt.bar(
        range(top_n),
        transcript_metrics['detected'].values[:top_n],
        tick_label=transcript_metrics['sirv_transcript'].values[:top_n]
    )
    plt.xticks(rotation=90)
    plt.xlabel('SIRV Transcript')
    plt.ylabel('Detection Rate')
    plt.title('Detection Rate by SIRV Transcript (Top 20)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'detection_rate_by_transcript.png'), dpi=300)
    plt.close()


def generate_report(
    comparison_file: str,
    output_html: str,
    template_file: Optional[str] = None
) -> str:
    """
    Generate an HTML report from comparison results.
    
    Args:
        comparison_file: Path to comparison CSV file
        output_html: Path to output HTML file
        template_file: Path to HTML template file (optional)
        
    Returns:
        str: Path to generated HTML report
    """
    try:
        import jinja2
    except ImportError:
        logger.warning("jinja2 not installed, skipping HTML report generation")
        return None
    
    # Load comparison data
    comparison = pd.read_csv(comparison_file)
    
    # Generate summary
    summary = _generate_summary(comparison)
    
    # Create HTML report
    if template_file and os.path.exists(template_file):
        # Load template
        with open(template_file, 'r') as f:
            template_str = f.read()
    else:
        # Use default template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SIRV Integration Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <h1>SIRV Integration Evaluation Report</h1>
            
            <h2>Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Expected Transcripts</td><td>{{ summary.total_expected }}</td></tr>
                <tr><td>Detected Transcripts</td><td>{{ summary.total_detected }}</td></tr>
                <tr><td>Detection Rate</td><td>{{ "%.2f%%" | format(summary.detection_rate * 100) }}</td></tr>
                <tr><td>Correlation</td><td>{{ "%.4f" | format(summary.correlation) }}</td></tr>
            </table>
            
            <h2>Top 10 Transcripts by Detection Rate</h2>
            <table>
                <tr><th>Transcript</th><th>Expected Count</th><th>Observed Count</th><th>Detection Rate</th></tr>
                {% for _, row in transcript_metrics.head(10).iterrows() %}
                <tr>
                    <td>{{ row.sirv_transcript }}</td>
                    <td>{{ row.expected_count }}</td>
                    <td>{{ row.observed_count }}</td>
                    <td>{{ "%.2f%%" | format(row.detected * 100) }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Top 10 Cells by Detection Rate</h2>
            <table>
                <tr><th>Cell</th><th>Expected Count</th><th>Observed Count</th><th>Detection Rate</th></tr>
                {% for _, row in cell_metrics.head(10).iterrows() %}
                <tr>
                    <td>{{ row.barcode }}</td>
                    <td>{{ row.expected_count }}</td>
                    <td>{{ row.observed_count }}</td>
                    <td>{{ "%.2f%%" | format(row.detected * 100) }}</td>
                </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        """
    
    # Prepare data for template
    transcript_metrics = summary['transcript_metrics'].reset_index()
    transcript_metrics = transcript_metrics.sort_values('detected', ascending=False)
    
    cell_metrics = summary['cell_metrics'].reset_index()
    cell_metrics = cell_metrics.sort_values('detected', ascending=False)
    
    # Render template
    template = jinja2.Template(template_str)
    html = template.render(
        summary=summary,
        transcript_metrics=transcript_metrics,
        cell_metrics=cell_metrics
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
    
    # Write HTML report
    with open(output_html, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML report generated: {output_html}")
    
    return output_html