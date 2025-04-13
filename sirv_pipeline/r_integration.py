"""
R Integration module for the SIRV Integration Pipeline.

This module provides integration with R code using rpy2, particularly for 
calling BamSlam functions directly from Python without intermediate files.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
import tempfile
import traceback

# Initialize logger
logger = logging.getLogger(__name__)

# Try to import rpy2
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    
    # Enable pandas-to-R conversion
    pandas2ri.activate()
    
    # Import required R packages
    base = importr('base')
    stats = importr('stats')
    utils = importr('utils')
    
    # Try to import optional R packages
    try:
        ggplot2 = importr('ggplot2')
        dplyr = importr('dplyr')
        tidyr = importr('tidyr')
        GenomicAlignments = importr('GenomicAlignments')
        SummarizedExperiment = importr('SummarizedExperiment')
        S4Vectors = importr('S4Vectors')
        IRanges = importr('IRanges')
        BiocGenerics = importr('BiocGenerics')
        data_table = importr('data.table')
    except Exception as e:
        logger.warning(f"Some R packages could not be imported: {e}")
        logger.warning("Some BamSlam functionality may not be available")
        
    RLIB_AVAILABLE = True
    logger.info("rpy2 successfully imported")
    
except ImportError:
    logger.warning("rpy2 not available - R integration functionality will be disabled")
    RLIB_AVAILABLE = False


class BamSlamR:
    """
    Class for integrating with BamSlam R code.
    
    This class provides Python wrappers around BamSlam R functions, allowing
    for direct analysis of BAM files without intermediate files.
    """
    
    def __init__(self, bamslam_r_path=None):
        """
        Initialize the BamSlam R integration.
        
        Args:
            bamslam_r_path: Path to the BamSlam.R file. If None, will look for default locations.
        """
        self.logger = logging.getLogger(__name__)
        
        if not RLIB_AVAILABLE:
            self.logger.error("rpy2 not available - BamSlamR cannot be used")
            raise ImportError("rpy2 not available - BamSlamR cannot be used")
            
        # Find BamSlam.R
        if bamslam_r_path is None:
            # Try some default locations
            possible_paths = [
                "/data/gpfs/projects/punim2251/BamSlam/BamSlam.R",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../BamSlam/BamSlam.R"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../BamSlam/BamSlam.R"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "BamSlam.R"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    bamslam_r_path = path
                    break
                    
            if bamslam_r_path is None:
                self.logger.error("BamSlam.R not found - please specify path")
                raise FileNotFoundError("BamSlam.R not found - please specify path")
        
        self.bamslam_r_path = bamslam_r_path
        self.logger.info(f"Using BamSlam.R at: {self.bamslam_r_path}")
        
        # Source BamSlam.R
        self._source_bamslam()
        
    def _source_bamslam(self):
        """Source the BamSlam.R file."""
        try:
            # Source the R file
            ro.r(f'source("{self.bamslam_r_path}")')
            
            # Check if required functions are available
            required_functions = [
                "import_bam_file",
                "get_read_coverages",
                "get_alignment_data_and_summarise",
                "plot_coverage",
                "plot_coverage_vs_length"
            ]
            
            # Check if functions exist in R environment
            available_functions = ro.r('ls()')
            
            # Convert R object to Python list
            available_functions_list = [str(func) for func in available_functions]
            
            missing_functions = [func for func in required_functions if func not in available_functions_list]
            
            if missing_functions:
                self.logger.warning(f"Some required functions not found in BamSlam.R: {missing_functions}")
                self.logger.warning("Some functionality may not work properly")
            
            self.logger.info("BamSlam.R successfully sourced")
            
        except Exception as e:
            self.logger.error(f"Error sourcing BamSlam.R: {e}")
            traceback.print_exc()
            raise
    
    def analyze_bam(self, bam_file, output_prefix, seq_type="cdna", 
                    extract_read_coverages=True, get_best_alignments=True,
                    length_filtered=False, min_length=None, max_length=None,
                    return_dataframes=True):
        """
        Analyze a BAM file using BamSlam.
        
        Args:
            bam_file: Path to BAM file
            output_prefix: Prefix for output files
            seq_type: Sequencing type ('cdna' or 'rna')
            extract_read_coverages: Whether to extract read coverages
            get_best_alignments: Whether to get best alignments
            length_filtered: Whether to filter by transcript length
            min_length: Minimum transcript length
            max_length: Maximum transcript length
            return_dataframes: Whether to return pandas DataFrames
            
        Returns:
            Tuple of (bam_data, primary_alignments, transcript_data, stats)
        """
        
        if not os.path.exists(bam_file):
            self.logger.error(f"BAM file does not exist: {bam_file}")
            raise FileNotFoundError(f"BAM file does not exist: {bam_file}")
            
        self.logger.info(f"Analyzing BAM file: {bam_file}")
        
        try:
            # Call import_bam_file
            # Convert output path to R string
            bam_file_r = ro.r.file.path(bam_file)
            
            # Convert seq_type to R string
            seq_type_r = ro.StrVector([seq_type])
            
            # Import BAM file
            self.logger.info(f"Importing BAM file: {bam_file}")
            bam_data_r = ro.r['import_bam_file'](bam_file_r, seq_type_r)
            
            # Apply length filtering if requested
            if length_filtered and (min_length is not None or max_length is not None):
                self.logger.info(f"Filtering by transcript length: min={min_length}, max={max_length}")
                
                # Create R expression for filtering
                filter_expr = ""
                if min_length is not None:
                    filter_expr += f"seqlengths >= {min_length}"
                if max_length is not None:
                    if filter_expr:
                        filter_expr += " & "
                    filter_expr += f"seqlengths <= {max_length}"
                
                # Apply filter
                bam_data_r = ro.r(f"subset(bam_data_r, {filter_expr})")
            
            # Extract read coverages if requested
            if extract_read_coverages:
                self.logger.info("Extracting read coverages")
                output_prefix_r = ro.StrVector([output_prefix])
                result_r = ro.r['get_read_coverages'](bam_data_r, output_prefix_r)
                bam_data_r = result_r[0]  # First element is bam_data
            
            # Get best alignments if requested
            if get_best_alignments:
                self.logger.info("Getting best alignments")
                output_prefix_r = ro.StrVector([output_prefix])
                primary_alignments_r = ro.r['get_alignment_data_and_summarise'](bam_data_r, output_prefix_r)
            else:
                primary_alignments_r = None
            
            # Get transcripts data
            transcript_data_r = None
            if os.path.exists(f"{output_prefix}_transcript_level_data.csv"):
                transcript_data_r = ro.r(f"read.csv('{output_prefix}_transcript_level_data.csv')")
            
            # Get stats data
            stats_r = None
            if os.path.exists(f"{output_prefix}_stats.csv"):
                stats_r = ro.r(f"read.csv('{output_prefix}_stats.csv', header=FALSE)")
            
            # Convert to pandas if requested
            if return_dataframes:
                with localconverter(ro.default_converter + pandas2ri.converter):
                    bam_data_pd = ro.conversion.rpy2py(bam_data_r)
                    
                    if primary_alignments_r is not None:
                        primary_alignments_pd = ro.conversion.rpy2py(primary_alignments_r)
                    else:
                        primary_alignments_pd = None
                        
                    if transcript_data_r is not None:
                        transcript_data_pd = ro.conversion.rpy2py(transcript_data_r)
                    else:
                        transcript_data_pd = None
                        
                    if stats_r is not None:
                        stats_pd = ro.conversion.rpy2py(stats_r)
                        # Convert stats to dict for easier use
                        if len(stats_pd.columns) >= 2:
                            stats_dict = dict(zip(stats_pd.iloc[:, 0], stats_pd.iloc[:, 1]))
                        else:
                            stats_dict = {}
                    else:
                        stats_dict = {}
                
                return bam_data_pd, primary_alignments_pd, transcript_data_pd, stats_dict
            else:
                return bam_data_r, primary_alignments_r, transcript_data_r, stats_r
                
        except Exception as e:
            self.logger.error(f"Error analyzing BAM file: {e}")
            traceback.print_exc()
            raise
    
    def compare_datasets_by_length(self, 
                                   datasets: Dict[str, pd.DataFrame], 
                                   length_bins: Optional[List[Tuple[float, float]]] = None,
                                   metrics: Optional[List[str]] = None):
        """
        Compare datasets by transcript length, matching transcripts of similar lengths.
        
        This is more precise than the binning approach used in the current implementation.
        
        Args:
            datasets: Dictionary mapping dataset name to DataFrame with BAM data
            length_bins: List of (min, max) tuples for length bins. If None, will create automatically.
            metrics: List of metrics to compare. If None, will use defaults.
            
        Returns:
            DataFrame with comparative metrics across datasets by length bin
        """
        if not datasets:
            self.logger.error("No datasets provided for comparison")
            return None
            
        self.logger.info(f"Comparing {len(datasets)} datasets by transcript length")
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = [
                'read_coverage',
                'aligned_fraction',
                'read_accuracy'
            ]
        
        # Extract transcript lengths from all datasets
        all_transcript_lengths = []
        for name, df in datasets.items():
            # Check if transcript_length column exists
            if 'transcript_length' not in df.columns:
                self.logger.warning(f"Dataset {name} does not have transcript_length column")
                continue
                
            all_transcript_lengths.extend(df['transcript_length'].unique())
        
        # Create length bins if not provided
        if length_bins is None:
            # Calculate min, max, and percentiles
            min_length = min(all_transcript_lengths)
            max_length = max(all_transcript_lengths)
            percentiles = np.percentile(all_transcript_lengths, [0, 25, 50, 75, 100])
            
            # Create 4 bins based on percentiles
            length_bins = [
                (percentiles[0], percentiles[1]),  # 0-25%
                (percentiles[1], percentiles[2]),  # 25-50%
                (percentiles[2], percentiles[3]),  # 50-75%
                (percentiles[3], percentiles[4])   # 75-100%
            ]
            
            self.logger.info(f"Created {len(length_bins)} length bins: {length_bins}")
        
        # Initialize results dictionary
        results = {}
        
        # Process each length bin
        for i, (min_len, max_len) in enumerate(length_bins):
            bin_name = f"bin_{i}_({int(min_len)}-{int(max_len)})"
            self.logger.info(f"Processing length bin {bin_name}")
            
            bin_results = {}
            
            # Process each dataset
            for name, df in datasets.items():
                # Filter by transcript length
                if 'transcript_length' not in df.columns:
                    self.logger.warning(f"Dataset {name} does not have transcript_length column - skipping")
                    continue
                    
                # Filter to primary alignments if read_id is available
                if 'read_id' in df.columns:
                    df_primary = df.sort_values(['read_id', 'mapq'], ascending=[True, False]) \
                                  .groupby('read_id').first().reset_index()
                else:
                    df_primary = df
                
                # Filter by length
                df_filtered = df_primary[(df_primary['transcript_length'] >= min_len) & 
                                         (df_primary['transcript_length'] <= max_len)]
                
                if len(df_filtered) == 0:
                    self.logger.warning(f"No reads in dataset {name} for length bin {bin_name}")
                    bin_results[name] = {
                        'count': 0,
                        'transcript_length_mean': 0,
                        'transcript_length_median': 0
                    }
                    for metric in metrics:
                        if metric in df.columns:
                            bin_results[name][f"{metric}_mean"] = 0
                            bin_results[name][f"{metric}_median"] = 0
                            bin_results[name][f"{metric}_std"] = 0
                    continue
                
                # Calculate summary statistics
                bin_results[name] = {
                    'count': len(df_filtered),
                    'transcript_length_mean': df_filtered['transcript_length'].mean(),
                    'transcript_length_median': df_filtered['transcript_length'].median()
                }
                
                # Calculate statistics for each metric
                for metric in metrics:
                    if metric in df_filtered.columns:
                        bin_results[name][f"{metric}_mean"] = df_filtered[metric].mean()
                        bin_results[name][f"{metric}_median"] = df_filtered[metric].median()
                        bin_results[name][f"{metric}_std"] = df_filtered[metric].std()
            
            # Store results for this bin
            results[bin_name] = bin_results
        
        # Convert to DataFrame
        results_rows = []
        for bin_name, bin_results in results.items():
            for dataset_name, dataset_stats in bin_results.items():
                row = {
                    'length_bin': bin_name,
                    'dataset': dataset_name
                }
                row.update(dataset_stats)
                results_rows.append(row)
        
        return pd.DataFrame(results_rows)
    
    def plot_comparative_metrics_by_length(self,
                                         comparison_results: pd.DataFrame,
                                         metrics: List[str],
                                         output_file: str,
                                         title: str = "Comparative Metrics by Transcript Length",
                                         dataset_colors: Optional[Dict[str, str]] = None):
        """
        Plot comparative metrics by transcript length bin.
        
        Args:
            comparison_results: DataFrame from compare_datasets_by_length
            metrics: List of metrics to plot
            output_file: Path to output file
            title: Plot title
            dataset_colors: Dictionary mapping dataset names to colors
        """
        if comparison_results is None or comparison_results.empty:
            self.logger.error("No comparison results provided for plotting")
            return
            
        self.logger.info(f"Plotting comparative metrics by transcript length to {output_file}")
        
        # Check if metrics exist in the data
        valid_metrics = []
        for metric in metrics:
            metric_mean = f"{metric}_mean"
            if metric_mean in comparison_results.columns:
                valid_metrics.append(metric)
            else:
                self.logger.warning(f"Metric {metric} not found in comparison results - skipping")
        
        if not valid_metrics:
            self.logger.error("No valid metrics found for plotting")
            return
            
        # Get unique datasets and length bins
        datasets = comparison_results['dataset'].unique()
        length_bins = comparison_results['length_bin'].unique()
        
        # Set up colors if not provided
        if dataset_colors is None:
            # Use default color cycle
            default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            dataset_colors = {dataset: default_colors[i % len(default_colors)] 
                              for i, dataset in enumerate(datasets)}
        
        # Set up the figure
        n_metrics = len(valid_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5 * n_metrics), sharex=True)
        
        # Handle case with only one metric
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(valid_metrics):
            ax = axes[i]
            
            # Set up bar positions
            n_datasets = len(datasets)
            n_bins = len(length_bins)
            bar_width = 0.8 / n_datasets
            
            # Plot each dataset
            for j, dataset in enumerate(datasets):
                # Filter data for this dataset
                dataset_data = comparison_results[comparison_results['dataset'] == dataset]
                
                # Extract values
                x = np.arange(len(length_bins))
                
                # Map bin names to x positions
                bin_to_x = {bin_name: idx for idx, bin_name in enumerate(length_bins)}
                
                # Get values for each bin
                heights = []
                errors = []
                x_pos = []
                
                for bin_name in length_bins:
                    bin_data = dataset_data[dataset_data['length_bin'] == bin_name]
                    if not bin_data.empty:
                        heights.append(bin_data[f"{metric}_mean"].values[0])
                        errors.append(bin_data[f"{metric}_std"].values[0] if f"{metric}_std" in bin_data.columns else 0)
                        x_pos.append(bin_to_x[bin_name])
                    else:
                        heights.append(0)
                        errors.append(0)
                        x_pos.append(bin_to_x[bin_name])
                
                # Plot bars
                positions = [pos + j * bar_width - (n_datasets - 1) * bar_width / 2 for pos in x_pos]
                ax.bar(positions, heights, width=bar_width, 
                       color=dataset_colors[dataset], label=dataset,
                       yerr=errors, capsize=3)
            
            # Set labels and title
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f"{metric.replace('_', ' ').title()} by Transcript Length")
            ax.set_xticks(x)
            ax.set_xticklabels([bin_name.replace('bin_', 'Bin ').split('_')[0] for bin_name in length_bins],
                               rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add text labels
            for j, dataset in enumerate(datasets):
                dataset_data = comparison_results[comparison_results['dataset'] == dataset]
                for bin_name in length_bins:
                    bin_data = dataset_data[dataset_data['length_bin'] == bin_name]
                    if not bin_data.empty and bin_name in bin_to_x:
                        x_pos = bin_to_x[bin_name] + j * bar_width - (n_datasets - 1) * bar_width / 2
                        height = bin_data[f"{metric}_mean"].values[0]
                        count = bin_data['count'].values[0]
                        ax.text(x_pos, height, f"n={count}", ha='center', va='bottom', fontsize=8, 
                                rotation=90, color='black')
        
        # Add overall title
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Saved comparative metrics plot to {output_file}")
        
    def match_datasets_by_transcript_length(self,
                                          primary_datasets: Dict[str, pd.DataFrame],
                                          reference_dataset: str,
                                          tolerance: float = 0.1,
                                          min_count: int = 5,
                                          metrics: Optional[List[str]] = None):
        """
        Match transcripts from different datasets based on similar lengths.
        
        This is a more precise approach than binning, as it directly matches
        transcripts with similar lengths between datasets.
        
        Args:
            primary_datasets: Dictionary mapping dataset name to DataFrame with primary alignments
            reference_dataset: Key in primary_datasets to use as reference
            tolerance: Fractional difference in length allowed for matching
            min_count: Minimum number of transcripts required for comparison
            metrics: List of metrics to compare. If None, will use defaults.
            
        Returns:
            DataFrame with matched transcripts and comparative metrics
        """
        if reference_dataset not in primary_datasets:
            self.logger.error(f"Reference dataset {reference_dataset} not found")
            return None
            
        self.logger.info(f"Matching datasets by transcript length, using {reference_dataset} as reference")
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = [
                'read_coverage',
                'aligned_fraction',
                'read_accuracy'
            ]
        
        # Get reference transcripts
        ref_df = primary_datasets[reference_dataset]
        
        # Check if transcript_length and transcript_id columns exist
        if 'transcript_length' not in ref_df.columns or 'transcript_id' not in ref_df.columns:
            self.logger.error(f"Reference dataset missing required columns")
            return None
            
        # Group by transcript_id to get unique transcripts
        ref_transcripts = ref_df.groupby('transcript_id').agg({
            'transcript_length': 'first',
            'read_id': 'count'
        }).reset_index()
        
        ref_transcripts = ref_transcripts.rename(columns={'read_id': 'count'})
        
        # Filter to transcripts with sufficient read coverage
        ref_transcripts = ref_transcripts[ref_transcripts['count'] >= min_count]
        
        if len(ref_transcripts) == 0:
            self.logger.error(f"No reference transcripts with sufficient read coverage")
            return None
            
        self.logger.info(f"Found {len(ref_transcripts)} reference transcripts with >= {min_count} reads")
        
        # Initialize results
        matched_results = []
        
        # Process each reference transcript
        for _, ref_row in ref_transcripts.iterrows():
            ref_id = ref_row['transcript_id']
            ref_length = ref_row['transcript_length']
            
            # Initialize comparison row
            comparison = {
                'transcript_id': ref_id,
                'transcript_length': ref_length,
                f"{reference_dataset}_count": ref_row['count']
            }
            
            # Get reference reads for this transcript
            ref_reads = ref_df[ref_df['transcript_id'] == ref_id]
            
            # Calculate reference metrics
            for metric in metrics:
                if metric in ref_reads.columns:
                    comparison[f"{reference_dataset}_{metric}_mean"] = ref_reads[metric].mean()
                    comparison[f"{reference_dataset}_{metric}_median"] = ref_reads[metric].median()
                    comparison[f"{reference_dataset}_{metric}_std"] = ref_reads[metric].std()
            
            # Process other datasets
            for dataset_name, dataset_df in primary_datasets.items():
                if dataset_name == reference_dataset:
                    continue
                    
                # Check if required columns exist
                if 'transcript_length' not in dataset_df.columns or 'transcript_id' not in dataset_df.columns:
                    self.logger.warning(f"Dataset {dataset_name} missing required columns - skipping")
                    continue
                
                # Find matching transcripts by length
                min_length = ref_length * (1 - tolerance)
                max_length = ref_length * (1 + tolerance)
                
                # Filter by length
                matching_transcripts = dataset_df[(dataset_df['transcript_length'] >= min_length) & 
                                                (dataset_df['transcript_length'] <= max_length)]
                
                # Count reads per transcript
                transcript_counts = matching_transcripts.groupby('transcript_id').size().reset_index(name='count')
                transcript_counts = transcript_counts[transcript_counts['count'] >= min_count]
                
                if len(transcript_counts) == 0:
                    self.logger.debug(f"No matching transcripts found in {dataset_name} for {ref_id} (length={ref_length})")
                    comparison[f"{dataset_name}_count"] = 0
                    for metric in metrics:
                        if metric in dataset_df.columns:
                            comparison[f"{dataset_name}_{metric}_mean"] = None
                            comparison[f"{dataset_name}_{metric}_median"] = None
                            comparison[f"{dataset_name}_{metric}_std"] = None
                    continue
                
                # Get best matching transcript (closest in length)
                matching_transcripts = matching_transcripts.merge(transcript_counts, on='transcript_id')
                
                # Group by transcript_id and calculate average length
                transcript_info = matching_transcripts.groupby('transcript_id').agg({
                    'transcript_length': 'first',
                    'count': 'first'
                }).reset_index()
                
                # Calculate length difference from reference
                transcript_info['length_diff'] = abs(transcript_info['transcript_length'] - ref_length)
                
                # Sort by length difference and get best match
                best_match = transcript_info.sort_values('length_diff').iloc[0]
                best_id = best_match['transcript_id']
                
                # Get reads for best matching transcript
                best_reads = matching_transcripts[matching_transcripts['transcript_id'] == best_id]
                
                # Store match info
                comparison[f"{dataset_name}_transcript_id"] = best_id
                comparison[f"{dataset_name}_transcript_length"] = best_match['transcript_length']
                comparison[f"{dataset_name}_count"] = best_match['count']
                
                # Calculate metrics
                for metric in metrics:
                    if metric in best_reads.columns:
                        comparison[f"{dataset_name}_{metric}_mean"] = best_reads[metric].mean()
                        comparison[f"{dataset_name}_{metric}_median"] = best_reads[metric].median()
                        comparison[f"{dataset_name}_{metric}_std"] = best_reads[metric].std()
            
            # Add to results
            matched_results.append(comparison)
        
        # Convert to DataFrame
        return pd.DataFrame(matched_results) 