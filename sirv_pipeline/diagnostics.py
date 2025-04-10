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

def plot_read_position_distribution(coverage_model, plots_dir: Path) -> List[Path]:
    """
    Plot the distribution of read positions along transcripts.
    Inspired by BamSlam's coverage visualization.
    
    Args:
        coverage_model: Coverage bias model object
        plots_dir: Directory to save plots
        
    Returns:
        List of paths to generated plot files
    """
    setup_plotting_style()
    output_files = []
    
    # Plot overall position distribution
    if coverage_model.position_distribution is not None:
        plt.figure(figsize=(10, 6))
        x, y = coverage_model.position_distribution
        plt.plot(x, y, linewidth=2)
        plt.xlabel('Relative Position in Transcript (5\' → 3\')')
        plt.ylabel('Density')
        plt.title('Read Start Position Bias Across Transcripts')
        plt.xlim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add annotations to highlight 5' and 3' ends
        plt.annotate('5\' End', xy=(0.05, 0.05), xycoords='axes fraction', fontsize=12)
        plt.annotate('3\' End', xy=(0.95, 0.05), xycoords='axes fraction', fontsize=12, ha='right')
        
        # Save figure
        output_file = plots_dir / "overall_position_distribution.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        output_files.append(output_file)
    
    # Plot length-dependent position distributions if available
    if hasattr(coverage_model, 'length_dependent_distributions') and coverage_model.length_dependent_distributions:
        plt.figure(figsize=(12, 8))
        
        for i, (min_len, max_len) in enumerate(zip(coverage_model.length_bins[:-1], coverage_model.length_bins[1:])):
            if i in coverage_model.length_dependent_distributions:
                x, y = coverage_model.length_dependent_distributions[i]
                plt.plot(x, y, linewidth=2, label=f'{int(min_len)}-{int(max_len)} bp')
        
        plt.xlabel('Relative Position in Transcript (5\' → 3\')')
        plt.ylabel('Density')
        plt.title('Length-Dependent Read Start Position Bias')
        plt.xlim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Transcript Length')
        
        # Add annotations to highlight 5' and 3' ends
        plt.annotate('5\' End', xy=(0.05, 0.05), xycoords='axes fraction', fontsize=12)
        plt.annotate('3\' End', xy=(0.95, 0.05), xycoords='axes fraction', fontsize=12, ha='right')
        
        # Save figure
        output_file = plots_dir / "length_dependent_position_distribution.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        output_files.append(output_file)
    
    return output_files

def plot_coverage_heatmap(coverage_model, bam_file, gtf_file, plots_dir: Path, max_transcripts=50) -> Path:
    """
    Create a heatmap of transcript coverage across selected transcripts.
    Inspired by BamSlam's coverage visualization.
    
    Args:
        coverage_model: Coverage bias model object
        bam_file: Path to BAM file with aligned reads
        gtf_file: Path to GTF/GFF annotation file
        plots_dir: Directory to save plots
        max_transcripts: Maximum number of transcripts to include
        
    Returns:
        Path to generated plot file
    """
    import pysam
    import numpy as np
    
    setup_plotting_style()
    
    # Parse transcript info from GTF
    transcripts = {}
    
    try:
        with open(gtf_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if fields[2] == 'transcript':
                    transcript_id = fields[8].split(';')[0].split('=')[1]
                    strand = fields[6]
                    length = int(fields[4])
                    transcripts[transcript_id] = {'strand': strand, 'length': length}
    except Exception as e:
        print(f"Error parsing GTF file: {e}")
        return None
    
    # Select transcripts to analyze
    transcript_read_counts = {}
    selected_transcripts = []
    
    try:
        # Count reads per transcript
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for transcript_id in transcripts:
                # Get read count for transcript
                count = bam.count(reference=transcript_id)
                transcript_read_counts[transcript_id] = count
                
        # Select top transcripts by read count
        sorted_transcripts = sorted(transcript_read_counts.items(), 
                                   key=lambda x: x[1], reverse=True)[:max_transcripts]
        selected_transcripts = [t for t, _ in sorted_transcripts if t in transcripts]
        
    except Exception as e:
        print(f"Error analyzing BAM file: {e}")
        # If BAM analysis fails, just take first few transcripts
        selected_transcripts = list(transcripts.keys())[:max_transcripts]
    
    # Cap at max_transcripts
    selected_transcripts = selected_transcripts[:max_transcripts]
    
    if not selected_transcripts:
        print("No transcripts selected for coverage heatmap")
        return None
    
    # Calculate coverage for each transcript
    num_bins = 100
    coverage_matrix = np.zeros((len(selected_transcripts), num_bins))
    
    try:
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for i, transcript_id in enumerate(selected_transcripts):
                transcript_length = transcripts[transcript_id]['length']
                counts = np.zeros(transcript_length)
                
                # Get coverage for each position
                for read in bam.fetch(transcript_id):
                    start = read.reference_start
                    end = read.reference_end or (start + read.query_length)
                    
                    # Make sure positions are valid
                    start = max(0, min(start, transcript_length-1))
                    end = max(0, min(end, transcript_length))
                    
                    # Add coverage
                    counts[start:end] += 1
                
                # Adjust strand if needed
                if transcripts[transcript_id]['strand'] == '-':
                    counts = counts[::-1]
                
                # Bin the counts
                binned_counts = np.zeros(num_bins)
                for j in range(num_bins):
                    bin_start = int(j * transcript_length / num_bins)
                    bin_end = int((j + 1) * transcript_length / num_bins)
                    if bin_start < bin_end:
                        binned_counts[j] = np.mean(counts[bin_start:bin_end])
                
                # Normalize by max count
                max_count = np.max(binned_counts)
                if max_count > 0:
                    binned_counts = binned_counts / max_count
                
                coverage_matrix[i, :] = binned_counts
                
    except Exception as e:
        print(f"Error creating coverage matrix: {e}")
        return None
    
    # Create heatmap
    plt.figure(figsize=(12, max(8, len(selected_transcripts) * 0.3)))
    
    # Use transcript ID and length for y-axis labels
    y_labels = [f"{t} ({transcripts[t]['length']} bp)" 
               for t in selected_transcripts]
    
    if HAS_SEABORN:
        sns.heatmap(coverage_matrix, cmap='viridis', yticklabels=y_labels)
    else:
        plt.imshow(coverage_matrix, aspect='auto', cmap='viridis')
        plt.yticks(range(len(y_labels)), y_labels)
    
    plt.xlabel('Relative Position in Transcript (5\' → 3\')')
    plt.title('Normalized Coverage Across Transcripts')
    
    # Add annotations to highlight 5' and 3' ends
    plt.annotate('5\' End', xy=(0, -0.05), xycoords='axes fraction', fontsize=12)
    plt.annotate('3\' End', xy=(1, -0.05), xycoords='axes fraction', fontsize=12, ha='right')
    
    # Save figure
    output_file = plots_dir / "transcript_coverage_heatmap.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return output_file

def plot_3prime_vs_5prime_bias(coverage_model, plots_dir: Path) -> Path:
    """
    Visualize 3' vs 5' bias in read coverage.
    Inspired by SimReadTruncs and BamSlam position analysis.
    
    Args:
        coverage_model: Coverage bias model object
        plots_dir: Directory to save plots
        
    Returns:
        Path to generated plot file
    """
    setup_plotting_style()
    
    # Skip if no position distribution
    if coverage_model.position_distribution is None:
        return None
    
    plt.figure(figsize=(10, 8))
    
    # Get the position distribution
    x, y = coverage_model.position_distribution
    
    # Divide transcript into 5' and 3' halves
    midpoint_idx = len(x) // 2
    prime5_area = np.sum(y[:midpoint_idx]) / np.sum(y)
    prime3_area = np.sum(y[midpoint_idx:]) / np.sum(y)
    
    # Plot the distribution with 5' and 3' halves highlighted
    plt.fill_between(x[:midpoint_idx], y[:midpoint_idx], alpha=0.4, color='blue', label=f"5\' Region ({prime5_area:.1%})")
    plt.fill_between(x[midpoint_idx:], y[midpoint_idx:], alpha=0.4, color='red', label=f"3\' Region ({prime3_area:.1%})")
    
    # Plot the line again on top for clarity
    plt.plot(x, y, 'k-', linewidth=1.5)
    
    # Calculate 5'/3' bias ratio (higher means more 3' biased)
    bias_ratio = prime3_area / prime5_area if prime5_area > 0 else float('inf')
    bias_text = f"3\'/5\' Bias Ratio: {bias_ratio:.2f}"
    
    bias_category = "balanced" if 0.8 <= bias_ratio <= 1.2 else \
                    "strongly 3\'-biased" if bias_ratio > 2 else \
                    "moderately 3\'-biased" if bias_ratio > 1.2 else \
                    "strongly 5\'-biased" if bias_ratio < 0.5 else "moderately 5\'-biased"
                    
    # Add annotations
    plt.annotate(bias_text + f"\nCoverage is {bias_category}", 
                xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    # Add vertical line at midpoint
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    
    plt.xlabel('Relative Position in Transcript (5\' → 3\')')
    plt.ylabel('Density')
    plt.title('5\' vs 3\' Coverage Bias Analysis')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    output_file = plots_dir / "5prime_vs_3prime_bias.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return output_file

def create_coverage_model_diagnostics(coverage_model, bam_file, gtf_file, plots_dir: Path) -> Dict[str, Path]:
    """
    Create comprehensive diagnostics for coverage model evaluation.
    Combines BamSlam-inspired visualizations with SimReadTruncs principles.
    
    Args:
        coverage_model: Coverage bias model object
        bam_file: Path to BAM file with aligned reads
        gtf_file: Path to GTF/GFF annotation file
        plots_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    plots = {}
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    # Position distribution plots
    position_plots = plot_read_position_distribution(coverage_model, plots_dir)
    for i, plot_path in enumerate(position_plots):
        plots[f"position_distribution_{i}"] = plot_path
    
    # Coverage heatmap
    heatmap_plot = plot_coverage_heatmap(coverage_model, bam_file, gtf_file, plots_dir)
    if heatmap_plot:
        plots["coverage_heatmap"] = heatmap_plot
    
    # 5' vs 3' bias analysis
    bias_plot = plot_3prime_vs_5prime_bias(coverage_model, plots_dir)
    if bias_plot:
        plots["prime_bias"] = bias_plot
    
    return plots

def create_diagnostic_visualizations(tracking_file: Path, 
                                   coverage_model_file: Path,
                                   eval_dir: Path,
                                   coverage_model=None,
                                   bam_file=None,
                                   gtf_file=None) -> Dict[str, Path]:
    """
    Create all diagnostic visualizations for the pipeline.
    
    Args:
        tracking_file: Path to tracking CSV file
        coverage_model_file: Path to coverage model CSV file
        eval_dir: Directory for evaluation outputs
        coverage_model: Coverage model object (optional)
        bam_file: Path to BAM file (optional)
        gtf_file: Path to GTF file (optional)
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    # Create output directories
    diagnostics_dir, plots_dir = create_output_dirs(eval_dir)
    
    # Dictionary to store plot file paths
    plots = {}
    
    # Read tracking data if file exists
    if tracking_file and tracking_file.exists():
        try:
            tracking_df = read_tracking_data(tracking_file)
            
            # Create tracking data visualizations
            plots['read_length'] = plot_read_length_distribution(tracking_df, plots_dir)
            plots['length_relationship'] = plot_length_sampling_relationship(tracking_df, plots_dir)
            plots['truncation'] = plot_truncation_distribution(tracking_df, plots_dir)
            plots['per_transcript'] = plot_per_transcript_length_distribution(tracking_df, plots_dir)
            plots['per_cell'] = plot_per_cell_integration_rate(tracking_df, plots_dir)
        except Exception as e:
            print(f"Error creating tracking visualizations: {e}")
    
    # Read coverage model data if file exists
    if coverage_model_file and coverage_model_file.exists():
        try:
            coverage_model_df = read_coverage_model(coverage_model_file)
            plots['coverage_model'] = plot_coverage_bias_model(coverage_model_df, plots_dir)
        except Exception as e:
            print(f"Error creating coverage model visualization: {e}")
    
    # Add BamSlam-inspired coverage model diagnostics if model object is available
    if coverage_model:
        try:
            coverage_plots = create_coverage_model_diagnostics(
                coverage_model, bam_file, gtf_file, plots_dir)
            plots.update(coverage_plots)
        except Exception as e:
            print(f"Error creating coverage model diagnostics: {e}")
    
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
                <img src="{rel_plots['read_length']}" alt="Read Length Distribution">
            </div>
            
            <div class="plot">
                <h2>Original vs Modified Length Relationship</h2>
                <div class="plot-description">
                    <p>This scatter plot shows the relationship between original and sampled lengths, indicating how closely sampled lengths match originals.</p>
                </div>
                <img src="{rel_plots['length_relationship']}" alt="Length Sampling Relationship">
            </div>
            
            <div class="plot">
                <h2>Truncation Distribution</h2>
                <div class="plot-description">
                    <p>Shows the distribution of truncation amounts (bp) across transcripts, helping understand how reads are shortened.</p>
                </div>
                <img src="{rel_plots['truncation']}" alt="Truncation Distribution">
            </div>
            
            <div class="plot">
                <h2>Per-Transcript Length Distribution</h2>
                <div class="plot-description">
                    <p>Violin plots showing the distribution of sampled read lengths for each transcript, with original length means as red dots.</p>
                </div>
                <img src="{rel_plots['per_transcript']}" alt="Per-Transcript Length Distribution">
            </div>
            
            <div class="plot">
                <h2>Coverage Bias Model</h2>
                <div class="plot-description">
                    <p>Heatmap showing the coverage bias model across normalized transcript positions (0-100%), visualizing 5' or 3' bias.</p>
                </div>
                <img src="{rel_plots['coverage_model']}" alt="Coverage Bias Model">
            </div>
            
            <div class="plot">
                <h2>SIRV Integration Rate by Cell</h2>
                <div class="plot-description">
                    <p>Bar chart showing the number of unique SIRV transcripts integrated into each cell barcode.</p>
                </div>
                <img src="{rel_plots['per_cell']}" alt="SIRV Integration Rate by Cell">
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