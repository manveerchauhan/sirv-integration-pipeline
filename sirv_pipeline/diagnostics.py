#!/usr/bin/env python3
"""
Diagnostic visualizations for SIRV Integration Pipeline.
Generates additional plots for quality control and debugging.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Try to import seaborn, but provide fallbacks if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    # Create a minimal fallback for sns.histplot that uses matplotlib
    class SnsFallback:
        @staticmethod
        def histplot(data=None, x=None, hue=None, bins=30, alpha=0.7, multiple="stack"):
            if hue is not None:
                # For hue plots, create separate histograms
                categories = data[hue].unique()
                for i, category in enumerate(categories):
                    subset = data[data[hue] == category]
                    values = subset[x].values
                    plt.hist(values, bins=bins, alpha=alpha, label=category)
                plt.legend()
            else:
                # Simple histogram
                plt.hist(data[x].values, bins=bins, alpha=alpha)
                
        @staticmethod
        def violinplot(data=None, x=None, y=None):
            # As a fallback, create a boxplot instead
            plt.boxplot([data[data[x] == category][y].values 
                         for category in sorted(data[x].unique())],
                        labels=sorted(data[x].unique()))
            
        @staticmethod
        def set_palette(palette):
            pass
            
    sns = SnsFallback()

def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('default')
    if HAS_SEABORN:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("viridis")
        except:
            # If specific style is not available, use default
            print("Warning: Could not use seaborn whitegrid style, using default style instead")
            
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    
    # Try to set font family, but handle gracefully if not available
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'FreeSans', 'sans-serif']
    except:
        print("Warning: Could not set custom fonts, using defaults instead")

def read_tracking_data(tracking_file: Path) -> pd.DataFrame:
    """Read tracking data from the given file."""
    if not tracking_file.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_file}")
    
    return pd.read_csv(tracking_file)

def read_coverage_model(coverage_model_file: Path) -> pd.DataFrame:
    """Read coverage model data from the given file."""
    if not coverage_model_file.exists():
        raise FileNotFoundError(f"Coverage model file not found: {coverage_model_file}")
    
    return pd.read_csv(coverage_model_file)

def create_output_dirs(eval_dir: Path) -> Tuple[Path, Path]:
    """Create output directories for diagnostic plots."""
    diagnostics_dir = eval_dir / "diagnostics"
    plots_dir = diagnostics_dir / "plots"
    
    for dir_path in [diagnostics_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return diagnostics_dir, plots_dir

def plot_read_length_distribution(tracking_df: pd.DataFrame, plots_dir: Path) -> Path:
    """
    Plot the distribution of original vs sampled read lengths.
    Shows histograms comparing original scRNA-seq read lengths vs. modified SIRV read lengths.
    """
    plt.figure(figsize=(12, 7))
    
    # Extract length data
    original_lengths = tracking_df['original_length'].values
    sampled_lengths = tracking_df['sampled_length'].values
    
    # Create histograms
    bins = np.linspace(min(min(original_lengths), min(sampled_lengths)), 
                       max(max(original_lengths), max(sampled_lengths)), 
                       30)
    
    plt.hist(original_lengths, bins=bins, alpha=0.7, label='Original Length', color='blue')
    plt.hist(sampled_lengths, bins=bins, alpha=0.7, label='Sampled Length', color='orange')
    
    # Add details
    plt.axvline(np.mean(original_lengths), color='blue', linestyle='dashed', linewidth=1, 
                label=f'Mean Original: {np.mean(original_lengths):.1f}')
    plt.axvline(np.mean(sampled_lengths), color='orange', linestyle='dashed', linewidth=1, 
                label=f'Mean Sampled: {np.mean(sampled_lengths):.1f}')
    
    plt.xlabel('Read Length (bp)')
    plt.ylabel('Frequency')
    plt.title('Original vs Sampled Read Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_file = plots_dir / "read_length_distribution.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return output_file

def plot_length_sampling_relationship(tracking_df: pd.DataFrame, plots_dir: Path) -> Path:
    """
    Create a scatterplot showing the relationship between original and sampled lengths.
    """
    plt.figure(figsize=(10, 10))
    
    # Create scatterplot with hexbin for density
    plt.hexbin(tracking_df['original_length'], tracking_df['sampled_length'], 
              gridsize=30, cmap='viridis', mincnt=1)
    
    # Add diagonal line (y=x)
    max_len = max(tracking_df['original_length'].max(), tracking_df['sampled_length'].max())
    min_len = min(tracking_df['original_length'].min(), tracking_df['sampled_length'].min())
    plt.plot([min_len, max_len], [min_len, max_len], 'r--', alpha=0.7, label='y=x')
    
    # Calculate and show correlation
    corr = np.corrcoef(tracking_df['original_length'], tracking_df['sampled_length'])[0, 1]
    plt.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    plt.xlabel('Original Length (bp)')
    plt.ylabel('Sampled Length (bp)')
    plt.title('Original vs Sampled Read Length Relationship')
    plt.colorbar(label='Number of Reads')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_file = plots_dir / "length_sampling_relationship.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return output_file

def plot_truncation_distribution(tracking_df: pd.DataFrame, plots_dir: Path) -> Path:
    """
    Plot the distribution of truncation amounts across transcripts.
    Shows where truncations occur along transcripts (distance from 5' end).
    """
    # Calculate truncation amount
    tracking_df['truncation'] = tracking_df['original_length'] - tracking_df['sampled_length']
    tracking_df['truncation_ratio'] = tracking_df['truncation'] / tracking_df['original_length']
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Only include reads with truncation
    truncated_reads = tracking_df[tracking_df['truncation'] > 0]
    if len(truncated_reads) == 0:
        plt.text(0.5, 0.5, "No truncated reads found", 
                 ha='center', va='center', fontsize=14)
    else:
        # Main plot
        sns.histplot(data=truncated_reads, x='truncation', hue='sirv_transcript', 
                    bins=30, alpha=0.7, multiple='stack')
        
        # Add info about percentage of truncated reads
        pct_truncated = len(truncated_reads) / len(tracking_df) * 100
        plt.annotate(f'Truncated reads: {pct_truncated:.1f}% ({len(truncated_reads)}/{len(tracking_df)})', 
                    xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=12, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    plt.xlabel('Truncation Amount (bp)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Read Truncations')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_file = plots_dir / "truncation_distribution.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    # Also create a plot for truncation ratio
    plt.figure(figsize=(12, 8))
    if len(truncated_reads) == 0:
        plt.text(0.5, 0.5, "No truncated reads found", 
                 ha='center', va='center', fontsize=14)
    else:
        sns.histplot(data=truncated_reads, x='truncation_ratio', hue='sirv_transcript', 
                    bins=30, alpha=0.7, multiple='stack')
        plt.xlabel('Truncation Ratio (truncation/original_length)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Relative Read Truncations')
        plt.grid(True, alpha=0.3)
    
    # Save figure
    ratio_output_file = plots_dir / "truncation_ratio_distribution.png"
    plt.tight_layout()
    plt.savefig(ratio_output_file, dpi=150)
    plt.close()
    
    return output_file

def plot_per_transcript_length_distribution(tracking_df: pd.DataFrame, plots_dir: Path) -> Path:
    """
    Plot the distribution of read lengths for each transcript.
    Shows how read lengths vary by transcript.
    """
    plt.figure(figsize=(14, 8))
    
    # Create violin plot
    sns.violinplot(data=tracking_df, x='sirv_transcript', y='sampled_length')
    
    # Add original length means as points
    transcript_means = tracking_df.groupby('sirv_transcript')['original_length'].mean()
    plt.scatter(range(len(transcript_means)), transcript_means.values, color='red', 
               marker='o', s=50, label='Mean Original Length')
    
    plt.xlabel('SIRV Transcript')
    plt.ylabel('Read Length (bp)')
    plt.title('Read Length Distribution by Transcript')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_file = plots_dir / "per_transcript_length_distribution.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return output_file

def plot_coverage_bias_model(coverage_model_df: pd.DataFrame, plots_dir: Path) -> Path:
    """
    Plot the coverage bias model across transcript positions.
    Shows a metagene-style plot of coverage across normalized transcript positions (0-100%).
    """
    plt.figure(figsize=(14, 8))
    
    # Extract bin positions (columns) and transcripts (rows)
    bin_cols = [col for col in coverage_model_df.columns if col.startswith('bin_')]
    transcripts = coverage_model_df['transcript_id'].values
    
    # Convert to values array for plotting
    coverage_data = coverage_model_df[bin_cols].values
    
    # Plot heatmap
    im = plt.imshow(coverage_data, aspect='auto', cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, label='Coverage Bias Factor')
    
    # Add labels
    plt.yticks(range(len(transcripts)), transcripts)
    
    # X-axis shows relative position along transcript
    num_bins = len(bin_cols)
    plt.xticks(np.linspace(0, num_bins-1, 11), 
              [f"{int(x*100)}%" for x in np.linspace(0, 1, 11)])
    
    plt.xlabel('Relative Position Along Transcript (5\' to 3\')')
    plt.ylabel('Transcript')
    plt.title('Coverage Bias Model Across Transcript Positions')
    
    # Save figure
    output_file = plots_dir / "coverage_bias_model.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    # Also create a line plot
    plt.figure(figsize=(14, 8))
    
    # Extract positions for x-axis (percentage along transcript)
    positions = np.linspace(0, 100, len(bin_cols))
    
    # Plot each transcript
    for i, transcript in enumerate(transcripts):
        plt.plot(positions, coverage_data[i], label=transcript, linewidth=2)
    
    plt.xlabel('Relative Position Along Transcript (%)')
    plt.ylabel('Coverage Bias Factor')
    plt.title('Coverage Bias Model Across Transcript Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    line_output_file = plots_dir / "coverage_bias_lineplot.png"
    plt.tight_layout()
    plt.savefig(line_output_file, dpi=150)
    plt.close()
    
    return output_file

def plot_per_cell_integration_rate(tracking_df: pd.DataFrame, plots_dir: Path) -> Path:
    """
    Plot the integration rate for each cell barcode.
    Shows a bar chart of SIRVs integrated into each cell.
    """
    plt.figure(figsize=(14, 8))
    
    # Get counts per barcode
    barcode_counts = tracking_df.groupby('barcode')['sirv_transcript'].nunique().sort_values(ascending=False)
    
    # Plot bar chart
    bars = plt.bar(range(len(barcode_counts)), barcode_counts.values)
    
    # Add horizontal line at mean
    mean_count = barcode_counts.mean()
    plt.axhline(mean_count, color='red', linestyle='--', 
               label=f'Mean: {mean_count:.1f} SIRV transcripts per cell')
    
    # Add text annotation for statistics
    std_count = barcode_counts.std()
    plt.annotate(f'Mean: {mean_count:.1f}, Std: {std_count:.1f}\nRange: {barcode_counts.min()}-{barcode_counts.max()}', 
                xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    plt.xlabel('Cell Barcode')
    plt.ylabel('Number of Unique SIRV Transcripts')
    plt.title('SIRV Integration Rate by Cell')
    plt.xticks(range(len(barcode_counts)), barcode_counts.index, rotation=90)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_file = plots_dir / "per_cell_integration_rate.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    # Also create a histogram of SIRV transcripts per cell
    plt.figure(figsize=(10, 6))
    plt.hist(barcode_counts.values, bins=10, alpha=0.7, color='blue')
    plt.axvline(mean_count, color='red', linestyle='--', 
               label=f'Mean: {mean_count:.1f} SIRV transcripts per cell')
    
    plt.xlabel('Number of Unique SIRV Transcripts')
    plt.ylabel('Number of Cells')
    plt.title('Distribution of SIRV Transcripts per Cell')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    hist_output_file = plots_dir / "sirv_per_cell_histogram.png"
    plt.tight_layout()
    plt.savefig(hist_output_file, dpi=150)
    plt.close()
    
    return output_file

def create_diagnostic_visualizations(tracking_file: Path, 
                                    coverage_model_file: Path,
                                    eval_dir: Path) -> Dict[str, Path]:
    """
    Create diagnostic visualizations for the SIRV integration pipeline.
    
    Args:
        tracking_file: Path to the tracking CSV file
        coverage_model_file: Path to the coverage model CSV file
        eval_dir: Path to the evaluation directory where outputs will be saved
    
    Returns:
        Dictionary of plot names to file paths
    """
    # Set up plotting style
    setup_plotting_style()
    
    # Create output directories
    diagnostics_dir, plots_dir = create_output_dirs(eval_dir)
    
    # Read input data
    tracking_df = read_tracking_data(tracking_file)
    coverage_model_df = read_coverage_model(coverage_model_file)
    
    # Initialize output dictionary
    plots = {}
    
    # Generate plots
    plots['read_length_distribution'] = plot_read_length_distribution(tracking_df, plots_dir)
    plots['length_sampling_relationship'] = plot_length_sampling_relationship(tracking_df, plots_dir)
    plots['truncation_distribution'] = plot_truncation_distribution(tracking_df, plots_dir)
    plots['per_transcript_length_distribution'] = plot_per_transcript_length_distribution(tracking_df, plots_dir)
    plots['coverage_bias_model'] = plot_coverage_bias_model(coverage_model_df, plots_dir)
    plots['per_cell_integration_rate'] = plot_per_cell_integration_rate(tracking_df, plots_dir)
    
    # Create an HTML summary page
    html_file = create_diagnostics_html(diagnostics_dir, plots)
    
    return plots

def create_diagnostics_html(diagnostics_dir: Path, plots: Dict[str, Path]) -> Path:
    """
    Create an HTML page with diagnostic visualizations.
    
    Args:
        diagnostics_dir: Directory to save the HTML file
        plots: Dictionary of plot names to file paths
    
    Returns:
        Path to the created HTML file
    """
    output_file = diagnostics_dir / "diagnostics.html"
    
    # Create relative paths to plots
    rel_plots = {name: os.path.relpath(path, diagnostics_dir) for name, path in plots.items()}
    
    # Build HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SIRV Integration Pipeline Diagnostics</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .summary {{ margin: 20px 0; }}
            .plots {{ display: flex; flex-direction: column; }}
            .plot {{ margin: 20px 0; }}
            .plot img {{ max-width: 100%; height: auto; }}
            .plot-description {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>SIRV Integration Pipeline Diagnostics</h1>
        
        <div class="summary">
            <h2>Diagnostic Visualizations</h2>
            <p>These plots provide insights into how the SIRV integration pipeline is processing and modifying reads.</p>
        </div>
        
        <div class="plots">
            <div class="plot">
                <h2>Read Length Distribution</h2>
                <div class="plot-description">
                    <p>Comparison of original vs. sampled read lengths shows how the coverage model affects read length selection.</p>
                </div>
                <img src="{rel_plots['read_length_distribution']}" alt="Read Length Distribution">
            </div>
            
            <div class="plot">
                <h2>Original vs Modified Length Relationship</h2>
                <div class="plot-description">
                    <p>This scatter plot shows the relationship between original and sampled lengths, indicating how closely sampled lengths match originals.</p>
                </div>
                <img src="{rel_plots['length_sampling_relationship']}" alt="Length Sampling Relationship">
            </div>
            
            <div class="plot">
                <h2>Truncation Distribution</h2>
                <div class="plot-description">
                    <p>Shows the distribution of truncation amounts (bp) across transcripts, helping understand how reads are shortened.</p>
                </div>
                <img src="{rel_plots['truncation_distribution']}" alt="Truncation Distribution">
            </div>
            
            <div class="plot">
                <h2>Per-Transcript Length Distribution</h2>
                <div class="plot-description">
                    <p>Violin plots showing the distribution of sampled read lengths for each transcript, with original length means as red dots.</p>
                </div>
                <img src="{rel_plots['per_transcript_length_distribution']}" alt="Per-Transcript Length Distribution">
            </div>
            
            <div class="plot">
                <h2>Coverage Bias Model</h2>
                <div class="plot-description">
                    <p>Heatmap showing the coverage bias model across normalized transcript positions (0-100%), visualizing 5' or 3' bias.</p>
                </div>
                <img src="{rel_plots['coverage_bias_model']}" alt="Coverage Bias Model">
            </div>
            
            <div class="plot">
                <h2>SIRV Integration Rate by Cell</h2>
                <div class="plot-description">
                    <p>Bar chart showing the number of unique SIRV transcripts integrated into each cell barcode.</p>
                </div>
                <img src="{rel_plots['per_cell_integration_rate']}" alt="SIRV Integration Rate by Cell">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file

if __name__ == "__main__":
    # Example usage
    tracking_file = Path("test_data/output/tracking.csv")
    coverage_model_file = Path("test_data/output/coverage_model.csv")
    eval_dir = Path("test_data/evaluation")
    
    if tracking_file.exists() and coverage_model_file.exists():
        create_diagnostic_visualizations(tracking_file, coverage_model_file, eval_dir)
        print(f"Diagnostic visualizations created in {eval_dir}/diagnostics/")
    else:
        print("Input files not found. Run the pipeline first.") 