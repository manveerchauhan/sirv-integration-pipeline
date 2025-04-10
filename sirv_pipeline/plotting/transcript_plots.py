"""
Transcript plotting functions for the SIRV integration pipeline.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def plot_transcript_counts(transcript_counts, output_file=None, top_n=20):
    """Plot transcript counts.
    
    Args:
        transcript_counts (dict): Dictionary of transcript counts
        output_file (str): Path to save the plot (optional)
        top_n (int): Number of top transcripts to show
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Convert to DataFrame if needed
        if isinstance(transcript_counts, dict):
            df = pd.DataFrame(list(transcript_counts.items()), 
                             columns=['transcript_id', 'count'])
        elif isinstance(transcript_counts, pd.DataFrame):
            # Ensure it has the right columns
            if 'transcript_id' not in transcript_counts.columns or \
               'count' not in transcript_counts.columns:
                logger.error("DataFrame must have 'transcript_id' and 'count' columns")
                return None
            df = transcript_counts
        else:
            logger.error(f"Unsupported type for transcript_counts: {type(transcript_counts)}")
            return None
        
        # Sort by count and take top N
        df_sorted = df.sort_values('count', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create horizontal bar plot
        ax.barh(df_sorted['transcript_id'], df_sorted['count'])
        
        # Add labels and title
        ax.set_xlabel('Count')
        ax.set_ylabel('Transcript ID')
        ax.set_title(f'Top {top_n} Transcript Counts')
        
        # Add count labels to bars
        for i, count in enumerate(df_sorted['count']):
            ax.text(count + max(df_sorted['count'])*0.01, i, str(count), 
                   verticalalignment='center')
        
        # Invert y-axis to show highest counts at the top
        ax.invert_yaxis()
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure if output file is specified
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved transcript counts plot to {output_file}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating transcript counts plot: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def plot_transcript_length_distribution(transcript_info, output_file=None, bins=30):
    """Plot transcript length distribution.
    
    Args:
        transcript_info (dict or DataFrame): Dictionary of transcript info with lengths or DataFrame
        output_file (str): Path to save the plot (optional)
        bins (int): Number of bins for histogram
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Extract transcript lengths
        if isinstance(transcript_info, dict):
            lengths = [info['length'] if isinstance(info, dict) else info 
                      for info in transcript_info.values()]
        elif isinstance(transcript_info, pd.DataFrame):
            # Check if DataFrame has length column
            if 'length' not in transcript_info.columns:
                logger.error("DataFrame must have 'length' column")
                return None
            lengths = transcript_info['length'].values
        else:
            logger.error(f"Unsupported type for transcript_info: {type(transcript_info)}")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        ax.hist(lengths, bins=bins, alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Transcript Length (bp)')
        ax.set_ylabel('Count')
        ax.set_title('Transcript Length Distribution')
        
        # Add statistics
        mean_length = np.mean(lengths)
        median_length = np.median(lengths)
        min_length = np.min(lengths)
        max_length = np.max(lengths)
        
        stats_text = f"Mean: {mean_length:.1f} bp\n" \
                    f"Median: {median_length:.1f} bp\n" \
                    f"Min: {min_length:.1f} bp\n" \
                    f"Max: {max_length:.1f} bp\n" \
                    f"Count: {len(lengths)}"
        
        # Add text box with statistics
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure if output file is specified
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved transcript length distribution plot to {output_file}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating transcript length distribution plot: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None 