"""
BamSlam-style visualization module for SIRV integration pipeline.

This module implements the visualization approaches from BamSlam for
analyzing SIRV alignment data, with a focus on coverage visualization.
"""

import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tempfile
import pysam
from typing import Dict, List, Optional, Tuple, Any, Union
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

def import_bam_file(bam_file: str, output_dir: str = None, cdna_mode: bool = True) -> pd.DataFrame:
    """
    Import BAM file and extract alignment metrics (BamSlam-inspired).

    Args:
        bam_file (str): Path to BAM file
        output_dir (str): Path to output directory (optional)
        cdna_mode (bool): Whether to filter for cDNA-specific flags

    Returns:
        pd.DataFrame: DataFrame with alignment metrics
    """
    logger.info(f"Importing BAM file: {bam_file}")
    
    try:
        # Open BAM file
        bam = pysam.AlignmentFile(bam_file, "rb")
        
        # Collect alignment data
        data = []
        
        # Count to track progress
        count = 0
        
        # Process all reads in BAM file
        for read in bam.fetch(until_eof=True):
            count += 1
            if count % 100000 == 0:
                logger.info(f"Processed {count:,} reads")
            
            # Skip unmapped reads
            if read.is_unmapped:
                continue
                
            # Filter reads based on flags (similar to BamSlam)
            if cdna_mode and not (read.flag in [0, 16, 256, 272]):
                continue
                
            # Extract alignment metrics
            try:
                # Get reference name (string) from reference_id (integer)
                ref_name = read.reference_name
                
                # Get reference length using the string name
                ref_length = bam.get_reference_length(ref_name)
                
                record = {
                    'read_id': read.query_name,
                    'transcript_id': ref_name,
                    'start': read.reference_start,
                    'end': read.reference_end,
                    'flag': read.flag,
                    'mapq': read.mapping_quality,
                    'transcript_length': ref_length,
                    'aligned_length': read.query_alignment_length,
                    'read_length': read.query_length
                }
                
                # Extract CIGAR information
                cigar_stats = read.get_cigar_stats()[0]
                record['M'] = cigar_stats[0]  # matches
                record['I'] = cigar_stats[1]  # insertions
                record['D'] = cigar_stats[2]  # deletions
                record['S'] = cigar_stats[4]  # soft clips
                record['H'] = cigar_stats[5]  # hard clips
                
                # Calculate metrics
                record['aligned_fraction'] = record['aligned_length'] / record['read_length'] if record['read_length'] > 0 else 0
                record['read_coverage'] = (record['end'] - record['start']) / record['transcript_length'] if record['transcript_length'] > 0 else 0
                
                # Extract NM tag (edit distance) if available
                try:
                    record['NM'] = read.get_tag('NM')
                    divisor = (record['M'] + record['I'] + record['D'])
                    record['read_accuracy'] = (divisor - record['NM']) / divisor if divisor > 0 else 0
                except KeyError:
                    record['NM'] = np.nan
                    record['read_accuracy'] = np.nan
                    
                # Add to data collection
                data.append(record)
            except Exception as e:
                # Skip reads with issues
                logger.debug(f"Skipping read {read.query_name}: {str(e)}")
        
        bam.close()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        logger.info(f"Imported {len(df)} alignments from BAM file")
        
        # Save data to CSV if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "bamslam_data.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved alignment data to {csv_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error importing BAM file: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return pd.DataFrame()

def summarize_alignment_data(data: pd.DataFrame, output_dir: str = None) -> Dict[str, Any]:
    """
    Create summary statistics for alignment data (BamSlam-inspired).

    Args:
        data (pd.DataFrame): Alignment data from import_bam_file
        output_dir (str): Path to output directory (optional)

    Returns:
        Dict[str, Any]: Summary statistics
    """
    logger.info("Generating alignment summary statistics")
    
    try:
        # Get primary alignments only (best score per read)
        primary = data.sort_values(['read_id', 'mapq', 'aligned_fraction'], ascending=[True, False, False])
        primary = primary.groupby('read_id').first().reset_index()
        
        # Transcript-level statistics
        transcript_stats = primary.groupby('transcript_id').agg(
            read_coverage=('read_coverage', 'median'),
            transcript_length=('transcript_length', 'max'),
            read_count=('read_id', 'count')
        ).reset_index()
        
        # Overall statistics
        stats = {
            'total_reads': len(primary),
            'full_length_reads': sum(primary['read_coverage'] > 0.95),
            'full_length_percentage': sum(primary['read_coverage'] > 0.95) / len(primary) * 100 if len(primary) > 0 else 0,
            'median_coverage': primary['read_coverage'].median(),
            'median_aligned_length': primary['aligned_length'].median(),
            'median_accuracy': primary['read_accuracy'].median() * 100,
            'unique_transcripts': len(transcript_stats),
            'median_transcript_coverage': transcript_stats['read_coverage'].median(),
            'median_transcript_length': transcript_stats['transcript_length'].median()
        }
        
        # Save statistics to CSV if output directory is specified
        if output_dir:
            # Save overall stats
            stats_df = pd.DataFrame([(k, v) for k, v in stats.items()], columns=['metric', 'value'])
            stats_path = os.path.join(output_dir, "bamslam_stats.csv")
            stats_df.to_csv(stats_path, index=False)
            
            # Save transcript-level stats
            transcript_path = os.path.join(output_dir, "bamslam_transcript_stats.csv")
            transcript_stats.to_csv(transcript_path, index=False)
            
            logger.info(f"Saved summary statistics to {stats_path}")
            logger.info(f"Saved transcript statistics to {transcript_path}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error generating summary statistics: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}

def plot_coverage_fraction(data: pd.DataFrame, output_dir: str = None) -> plt.Figure:
    """
    Plot read coverage fraction histogram (BamSlam-inspired).

    Args:
        data (pd.DataFrame): Alignment data from import_bam_file
        output_dir (str): Path to output directory (optional)

    Returns:
        plt.Figure: The generated figure
    """
    logger.info("Creating coverage fraction plot")
    
    try:
        # Get primary alignments only
        primary = data.sort_values(['read_id', 'mapq', 'aligned_fraction'], ascending=[True, False, False])
        primary = primary.groupby('read_id').first().reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Add 'above' column for coloring
        primary['above'] = primary['read_coverage'] > 0.95
        
        # Create histogram
        bins = np.linspace(0.5, 1, 181)  # 180 bins from 0.5 to 1
        ax.hist(primary['read_coverage'], bins=bins, color='gray', alpha=0.7)
        ax.hist(primary.loc[primary['above'], 'read_coverage'], bins=bins, color='steelblue')
        
        # Add vertical line at 0.95 (full-length cutoff)
        ax.axvline(x=0.95, color='black', linestyle='dashed', linewidth=0.5)
        
        # Add labels and title
        ax.set_xlabel("Coverage fraction")
        ax.set_ylabel("Number of reads")
        ax.set_xlim(0.5, 1)
        ax.grid(True, alpha=0.3)
        
        # Save plot if output directory is specified
        if output_dir:
            plot_path = os.path.join(output_dir, "coverage_fraction.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved coverage fraction plot to {plot_path}")
            
            # Also save PDF version
            plot_path_pdf = os.path.join(output_dir, "coverage_fraction.pdf")
            plt.savefig(plot_path_pdf, format='pdf', bbox_inches='tight')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating coverage fraction plot: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def plot_coverage_vs_length(data: pd.DataFrame, output_dir: str = None) -> plt.Figure:
    """
    Plot coverage fraction vs transcript length (BamSlam-inspired).

    Args:
        data (pd.DataFrame): Alignment data from import_bam_file
        output_dir (str): Path to output directory (optional)

    Returns:
        plt.Figure: The generated figure
    """
    logger.info("Creating coverage vs length plot")
    
    try:
        # Get primary alignments only
        primary = data.sort_values(['read_id', 'mapq', 'aligned_fraction'], ascending=[True, False, False])
        primary = primary.groupby('read_id').first().reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Create hexbin plot
        hb = ax.hexbin(
            primary['transcript_length'], 
            primary['read_coverage'], 
            gridsize=100, 
            cmap='viridis',
            bins='log',
            mincnt=1
        )
        
        # Add smoothed line showing the trend
        x_smooth = np.linspace(primary['transcript_length'].min(), min(primary['transcript_length'].max(), 15000), 100)
        try:
            # Use LOWESS smoother
            from statsmodels.nonparametric.smoothers_lowess import lowess
            lowess_data = lowess(
                primary['read_coverage'], 
                primary['transcript_length'], 
                frac=0.2, 
                it=3, 
                return_sorted=True
            )
            ax.plot(lowess_data[:, 0], lowess_data[:, 1], color='lavender', linewidth=2)
        except:
            # If statsmodels not available, use simple moving average
            bins = np.linspace(primary['transcript_length'].min(), primary['transcript_length'].max(), 20)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_means = []
            
            for i in range(len(bins) - 1):
                mask = (primary['transcript_length'] >= bins[i]) & (primary['transcript_length'] < bins[i+1])
                if sum(mask) > 0:
                    bin_means.append(primary.loc[mask, 'read_coverage'].mean())
                else:
                    bin_means.append(np.nan)
            
            # Remove NaN values
            valid = ~np.isnan(bin_means)
            ax.plot(bin_centers[valid], np.array(bin_means)[valid], color='lavender', linewidth=2)
        
        # Add colorbar
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('log(count)')
        
        # Add labels and title
        ax.set_xlabel("Known transcript length (nt)")
        ax.set_ylabel("Coverage fraction")
        ax.set_xlim(0, 15000)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Save plot if output directory is specified
        if output_dir:
            plot_path = os.path.join(output_dir, "coverage_vs_length.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved coverage vs length plot to {plot_path}")
            
            # Also save PDF version
            plot_path_pdf = os.path.join(output_dir, "coverage_vs_length.pdf")
            plt.savefig(plot_path_pdf, format='pdf', bbox_inches='tight')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating coverage vs length plot: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def plot_transcript_length_distribution(data: pd.DataFrame, output_dir: str = None) -> plt.Figure:
    """
    Plot transcript length distribution (BamSlam-inspired).

    Args:
        data (pd.DataFrame): Alignment data from import_bam_file
        output_dir (str): Path to output directory (optional)

    Returns:
        plt.Figure: The generated figure
    """
    logger.info("Creating transcript length distribution plot")
    
    try:
        # Get unique transcripts
        transcripts = data[['transcript_id', 'transcript_length']].drop_duplicates()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Create histogram
        bins = np.linspace(0, 10000, 180)  # 180 bins from 0 to 10000
        ax.hist(transcripts['transcript_length'], bins=bins, color='steelblue', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel("Known transcript length (nt)")
        ax.set_ylabel("Transcript count")
        ax.set_xlim(0, 10000)
        ax.grid(True, alpha=0.3)
        
        # Save plot if output directory is specified
        if output_dir:
            plot_path = os.path.join(output_dir, "transcript_length_distribution.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved transcript length distribution plot to {plot_path}")
            
            # Also save PDF version
            plot_path_pdf = os.path.join(output_dir, "transcript_length_distribution.pdf")
            plt.savefig(plot_path_pdf, format='pdf', bbox_inches='tight')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating transcript length distribution plot: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def plot_read_accuracy(data: pd.DataFrame, output_dir: str = None) -> plt.Figure:
    """
    Plot read accuracy distribution (BamSlam-inspired).

    Args:
        data (pd.DataFrame): Alignment data from import_bam_file
        output_dir (str): Path to output directory (optional)

    Returns:
        plt.Figure: The generated figure
    """
    logger.info("Creating read accuracy plot")
    
    try:
        # Get primary alignments only
        primary = data.sort_values(['read_id', 'mapq', 'aligned_fraction'], ascending=[True, False, False])
        primary = primary.groupby('read_id').first().reset_index()
        
        # Drop rows with NaN accuracy
        primary = primary.dropna(subset=['read_accuracy'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Create density plot
        sns.kdeplot(primary['read_accuracy'], ax=ax, color='steelblue', fill=True, alpha=0.4)
        
        # Add labels and title
        ax.set_xlabel("Read accuracy")
        ax.set_ylabel("Density")
        ax.set_xlim(0.5, 1)
        ax.grid(True, alpha=0.3)
        
        # Save plot if output directory is specified
        if output_dir:
            plot_path = os.path.join(output_dir, "read_accuracy.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved read accuracy plot to {plot_path}")
            
            # Also save PDF version
            plot_path_pdf = os.path.join(output_dir, "read_accuracy.pdf")
            plt.savefig(plot_path_pdf, format='pdf', bbox_inches='tight')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating read accuracy plot: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def run_bamslam_analysis(bam_file: str, output_dir: str, data_type: str = "cdna") -> Dict[str, Any]:
    """
    Run a complete BamSlam-style analysis on a BAM file.

    Args:
        bam_file (str): Path to BAM file
        output_dir (str): Path to output directory
        data_type (str): Data type, either "cdna" or "rna"

    Returns:
        Dict[str, Any]: Analysis results
    """
    logger.info(f"Running BamSlam analysis on {bam_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import BAM file
        data = import_bam_file(
            bam_file, 
            output_dir=output_dir, 
            cdna_mode=(data_type == "cdna")
        )
        
        if len(data) == 0:
            logger.error("No alignments found in BAM file")
            return {"success": False}
        
        # Generate summary statistics
        stats = summarize_alignment_data(data, output_dir=output_dir)
        
        # Create plots
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        coverage_fig = plot_coverage_fraction(data, output_dir=plots_dir)
        coverage_length_fig = plot_coverage_vs_length(data, output_dir=plots_dir)
        transcript_length_fig = plot_transcript_length_distribution(data, output_dir=plots_dir)
        accuracy_fig = plot_read_accuracy(data, output_dir=plots_dir)
        
        # Close figures to free memory
        plt.close(coverage_fig)
        plt.close(coverage_length_fig)
        plt.close(transcript_length_fig)
        plt.close(accuracy_fig)
        
        logger.info("BamSlam analysis completed successfully")
        
        return {
            "success": True,
            "stats": stats,
            "data_rows": len(data),
            "primary_reads": len(data.groupby('read_id'))
        }
        
    except Exception as e:
        logger.error(f"Error running BamSlam analysis: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return {"success": False, "error": str(e)} 