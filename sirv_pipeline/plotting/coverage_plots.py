"""
Coverage plotting functions for the SIRV integration pipeline.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def plot_coverage_bias(coverage_model, output_file=None):
    """Plot coverage bias from a coverage model.
    
    Args:
        coverage_model: The coverage bias model object
        output_file (str): Path to save the plot (optional)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # If the model has its own plotting method, use it
        if hasattr(coverage_model, 'plot_distributions'):
            return coverage_model.plot_distributions(output_file)
    except Exception as e:
        logger.error(f"Error using model's plot_distributions: {str(e)}")
    
    # Fallback to generic implementation
    try:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot coverage profile if available
        if hasattr(coverage_model, 'parameters') and 'profile' in coverage_model.parameters:
            profile = coverage_model.parameters['profile']
            x = np.linspace(0, 1, len(profile))
            ax1.plot(x, profile)
            ax1.set_xlabel("Relative position")
            ax1.set_ylabel("Relative coverage")
            ax1.set_title("Coverage bias profile")
            ax1.grid(True, alpha=0.3)
            
            # Add horizontal line at y=1 (no bias)
            ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # Plot length effect if available
        if hasattr(coverage_model, 'parameters') and 'length_effect' in coverage_model.parameters:
            length_effect = coverage_model.parameters['length_effect']
            if length_effect:
                groups = list(length_effect.keys())
                values = [length_effect[g] for g in groups]
                
                # Bar plot for length effect
                bar_positions = np.arange(len(groups))
                ax2.bar(bar_positions, values)
                ax2.set_xticks(bar_positions)
                ax2.set_xticklabels(groups, rotation=45)
                ax2.set_xlabel("Transcript length group")
                ax2.set_ylabel("Relative effect")
                ax2.set_title("Length effect on coverage")
                ax2.grid(True, alpha=0.3)
                
                # Add horizontal line at y=1 (no effect)
                ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # Add model type to title if available
        model_type = getattr(coverage_model, 'model_type', 'Unknown')
        fig.suptitle(f"Coverage Bias Model: {model_type}")
        plt.tight_layout()
        
        # Save figure if output file is specified
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved coverage bias plot to {output_file}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating coverage plot: {str(e)}")
        return None


def plot_transcript_coverage(bam_file, transcript_id, gtf_file=None, output_file=None):
    """Plot coverage for a specific transcript.
    
    Args:
        bam_file (str): Path to BAM file
        transcript_id (str): Transcript ID to plot
        gtf_file (str): Path to GTF file (optional)
        output_file (str): Path to save the plot (optional)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    import pysam
    
    try:
        # Open BAM file
        bam = pysam.AlignmentFile(bam_file, "rb")
        
        # Get transcript length
        transcript_length = None
        
        # Try to get from GTF if provided
        if gtf_file:
            try:
                import re
                with open(gtf_file, 'r') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue
                        
                        fields = line.strip().split('\t')
                        if len(fields) < 9:
                            continue
                        
                        if fields[2] != 'transcript':
                            continue
                        
                        # Check if this is our transcript
                        attr_str = fields[8]
                        tx_id_match = re.search(r'transcript_id "([^"]+)"', attr_str)
                        
                        if tx_id_match and tx_id_match.group(1) == transcript_id:
                            start = int(fields[3]) - 1  # 0-based
                            end = int(fields[4])
                            transcript_length = end - start
                            break
            except Exception as e:
                logger.warning(f"Error parsing GTF file: {str(e)}")
        
        # If we couldn't get length from GTF, try from BAM
        if transcript_length is None:
            try:
                for ref, length in zip(bam.references, bam.lengths):
                    if ref == transcript_id:
                        transcript_length = length
                        break
            except:
                pass
        
        # If we still don't have a length, use maximum position from reads
        if transcript_length is None:
            logger.warning(f"Could not determine length for transcript {transcript_id}, using maximum position from reads")
            max_pos = 0
            for read in bam.fetch(transcript_id):
                if read.reference_end > max_pos:
                    max_pos = read.reference_end
            transcript_length = max_pos
        
        # Initialize coverage array
        if transcript_length is None or transcript_length <= 0:
            logger.error(f"Invalid transcript length for {transcript_id}")
            return None
        
        coverage = np.zeros(transcript_length)
        
        # Calculate coverage
        for read in bam.fetch(transcript_id):
            for pos in range(read.reference_start, read.reference_end):
                if 0 <= pos < transcript_length:
                    coverage[pos] += 1
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(transcript_length), coverage)
        ax.set_xlabel("Position (bp)")
        ax.set_ylabel("Coverage (reads)")
        ax.set_title(f"Coverage for transcript {transcript_id}")
        ax.grid(True, alpha=0.3)
        
        # Add mean coverage line
        mean_coverage = np.mean(coverage)
        ax.axhline(y=mean_coverage, color='r', linestyle='--', alpha=0.5, 
                  label=f"Mean: {mean_coverage:.2f}")
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure if output file is specified
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved transcript coverage plot to {output_file}")
        
        bam.close()
        return fig
        
    except Exception as e:
        logger.error(f"Error creating transcript coverage plot: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None 