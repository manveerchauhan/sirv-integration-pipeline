"""
Plotting functions for comparative BamSlam analysis.

This module provides plotting functions for visualizing coverage patterns
and other metrics from BamSlam analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from pathlib import Path
import traceback

def plot_coverage_patterns(data, output_file, title="Coverage Pattern Comparison"):
    """
    Plot coverage patterns for comparative analysis.
    
    Args:
        data: Dictionary containing coverage data
        output_file: Path to output PNG file
        title: Plot title
    """
    logger = logging.getLogger()
    logger.info(f"Generating coverage pattern plot: {output_file}")
    
    try:
        # Extract data
        original_sirv = data.get('original_sirv', {}).get('coverage_data', pd.DataFrame())
        processed_sirv = data.get('processed_sirv', {}).get('coverage_data', pd.DataFrame())
        sc_data = data.get('sc', {}).get('coverage_data', pd.DataFrame())
        
        if original_sirv.empty or processed_sirv.empty or sc_data.empty:
            logger.warning("Missing coverage data for one or more datasets")
            return
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot original SIRV
        axes[0].plot(original_sirv['position'], original_sirv['coverage'], 
                    color='blue', alpha=0.7, linewidth=1.5)
        axes[0].set_title('Original SIRV Coverage')
        axes[0].set_ylabel('Coverage')
        
        # Plot processed SIRV
        axes[1].plot(processed_sirv['position'], processed_sirv['coverage'], 
                    color='green', alpha=0.7, linewidth=1.5)
        axes[1].set_title('Processed SIRV Coverage')
        axes[1].set_ylabel('Coverage')
        
        # Plot scRNA-seq
        axes[2].plot(sc_data['position'], sc_data['coverage'], 
                    color='red', alpha=0.7, linewidth=1.5)
        axes[2].set_title('scRNA-seq Coverage')
        axes[2].set_ylabel('Coverage')
        axes[2].set_xlabel('Normalized Transcript Position')
        
        # Set xlim
        for ax in axes:
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3)
        
        # Title and layout
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Coverage pattern plot saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating coverage pattern plot: {str(e)}")
        logger.error(traceback.format_exc())

def plot_length_distribution(data, output_file, title="Read Length Distribution"):
    """
    Plot read length distribution for comparative analysis.
    
    Args:
        data: Dictionary containing length data
        output_file: Path to output PNG file
        title: Plot title
    """
    logger = logging.getLogger()
    logger.info(f"Generating length distribution plot: {output_file}")
    
    try:
        # Extract data
        original_sirv = data.get('original_sirv', {}).get('length_data', pd.DataFrame())
        processed_sirv = data.get('processed_sirv', {}).get('length_data', pd.DataFrame())
        sc_data = data.get('sc', {}).get('length_data', pd.DataFrame())
        
        if original_sirv.empty or processed_sirv.empty or sc_data.empty:
            logger.warning("Missing length data for one or more datasets")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        bins = np.linspace(0, 5000, 50)
        
        ax.hist(original_sirv['read_length'], bins=bins, alpha=0.5, 
               label='Original SIRV', color='blue', density=True)
        ax.hist(processed_sirv['read_length'], bins=bins, alpha=0.5, 
               label='Processed SIRV', color='green', density=True)
        ax.hist(sc_data['read_length'], bins=bins, alpha=0.5, 
               label='scRNA-seq', color='red', density=True)
        
        ax.set_xlabel('Read Length (bp)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Length distribution plot saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating length distribution plot: {str(e)}")
        logger.error(traceback.format_exc())

def plot_coverage_bias(data, output_file, title="Coverage Bias Comparison"):
    """
    Plot coverage bias metrics for comparative analysis.
    
    Args:
        data: Dictionary containing coverage bias data
        output_file: Path to output PNG file
        title: Plot title
    """
    logger = logging.getLogger()
    logger.info(f"Generating coverage bias plot: {output_file}")
    
    try:
        # Extract metrics data
        metrics = {
            'Original SIRV': data.get('original_sirv', {}).get('metrics', {}),
            'Processed SIRV': data.get('processed_sirv', {}).get('metrics', {}),
            'scRNA-seq': data.get('sc', {}).get('metrics', {})
        }
        
        # Extract specific bias metrics
        bias_metrics = {
            '5\' Bias': {},
            '3\' Bias': {},
            'Coverage Evenness': {},
            'Coefficient of Variation': {}
        }
        
        for dataset, dataset_metrics in metrics.items():
            bias_metrics['5\' Bias'][dataset] = dataset_metrics.get('five_prime_bias', 0)
            bias_metrics['3\' Bias'][dataset] = dataset_metrics.get('three_prime_bias', 0)
            bias_metrics['Coverage Evenness'][dataset] = dataset_metrics.get('evenness', 0)
            bias_metrics['Coefficient of Variation'][dataset] = dataset_metrics.get('cv', 0)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, (metric_name, metric_data) in enumerate(bias_metrics.items()):
            datasets = list(metric_data.keys())
            values = list(metric_data.values())
            
            axes[i].bar(datasets, values, color=['blue', 'green', 'red'])
            axes[i].set_title(metric_name)
            axes[i].set_ylim(0, max(values) * 1.2)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for j, value in enumerate(values):
                axes[i].text(j, value + max(values) * 0.05, f'{value:.2f}', 
                           ha='center', va='bottom')
        
        # Adjust layout
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Coverage bias plot saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating coverage bias plot: {str(e)}")
        logger.error(traceback.format_exc())

def plot_length_stratified_coverage(data, output_file, title="Length-Stratified Coverage Comparison"):
    """
    Plot coverage patterns stratified by read length.
    
    Args:
        data: Dictionary containing length-stratified coverage data
        output_file: Path to output PNG file
        title: Plot title
    """
    logger = logging.getLogger()
    logger.info(f"Generating length-stratified coverage plot: {output_file}")
    
    try:
        # Extract data
        stratified_data = data.get('length_stratified', {})
        
        if not stratified_data:
            logger.warning("No length-stratified data available")
            return
        
        # Get datasets and length bins
        datasets = ['original_sirv', 'processed_sirv', 'sc']
        dataset_names = ['Original SIRV', 'Processed SIRV', 'scRNA-seq']
        dataset_colors = ['blue', 'green', 'red']
        
        # Get length bins
        length_bins = []
        for dataset in datasets:
            if dataset in stratified_data:
                length_bins = list(stratified_data[dataset].keys())
                break
        
        if not length_bins:
            logger.warning("No length bins found in stratified data")
            return
        
        # Create figure grid
        n_bins = len(length_bins)
        n_cols = 3
        n_rows = (n_bins + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, n_rows * 4))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # Plot each length bin
        for i, length_bin in enumerate(length_bins):
            row = i // n_cols
            col = i % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            
            # Plot each dataset
            for j, dataset in enumerate(datasets):
                if dataset in stratified_data and length_bin in stratified_data[dataset]:
                    bin_data = stratified_data[dataset][length_bin]
                    
                    if not bin_data.empty:
                        ax.plot(bin_data['position'], bin_data['coverage'], 
                               color=dataset_colors[j], alpha=0.7, linewidth=1.5,
                               label=dataset_names[j])
            
            ax.set_title(f'{length_bin}')
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3)
            
            # Only add x/y labels on the left/bottom
            if col == 0:
                ax.set_ylabel('Coverage')
            if row == n_rows - 1:
                ax.set_xlabel('Normalized Transcript Position')
        
        # Add legend to the figure
        handles, labels = [], []
        for j, dataset in enumerate(datasets):
            handles.append(plt.Line2D([0], [0], color=dataset_colors[j], linewidth=1.5))
            labels.append(dataset_names[j])
        
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                  ncol=3, fontsize=12)
        
        # Title and layout
        fig.suptitle(title, fontsize=16, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Length-stratified coverage plot saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating length-stratified coverage plot: {str(e)}")
        logger.error(traceback.format_exc())

def generate_comparison_heatmap(data, output_file, title="Similarity Heatmap"):
    """
    Generate a heatmap showing similarity between datasets.
    
    Args:
        data: Dictionary containing similarity data
        output_file: Path to output PNG file
        title: Plot title
    """
    logger = logging.getLogger()
    logger.info(f"Generating similarity heatmap: {output_file}")
    
    try:
        # Extract similarity matrix
        similarity = data.get('similarity_matrix', pd.DataFrame())
        
        if similarity.empty:
            logger.warning("No similarity matrix available")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Create heatmap
        sns.heatmap(similarity, annot=True, cmap='YlGnBu', vmin=0, vmax=1, 
                   square=True, linewidths=0.5, ax=ax)
        
        ax.set_title(title)
        
        # Save figure
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Similarity heatmap saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating similarity heatmap: {str(e)}")
        logger.error(traceback.format_exc()) 