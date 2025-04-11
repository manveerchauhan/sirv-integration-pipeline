"""
Plotting module for SIRV integration pipeline.

This module contains functions for creating plots and visualizations
for the SIRV integration pipeline.
"""

# Import BamSlam plots functionality
from sirv_pipeline.plotting.bamslam_plots import (
    run_bamslam_analysis,
    import_bam_file,
    plot_coverage_fraction,
    plot_coverage_vs_length,
    plot_transcript_length_distribution,
    plot_read_accuracy
)

def ensure_plotting_dirs(output_dir):
    """Ensure all plotting directories exist.
    
    Args:
        output_dir (str): Base output directory
    """
    import os
    
    # Create main plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create coverage plots directory
    coverage_dir = os.path.join(plots_dir, "coverage")
    os.makedirs(coverage_dir, exist_ok=True)
    
    # Create transcript plots directory
    transcript_dir = os.path.join(plots_dir, "transcripts")
    os.makedirs(transcript_dir, exist_ok=True)
    
    # Create bamslam plots directory
    bamslam_dir = os.path.join(plots_dir, "bamslam")
    os.makedirs(bamslam_dir, exist_ok=True)
    
    return {
        "base": plots_dir,
        "coverage": coverage_dir,
        "transcripts": transcript_dir,
        "bamslam": bamslam_dir
    } 