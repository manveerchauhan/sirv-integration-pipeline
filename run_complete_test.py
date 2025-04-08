#!/usr/bin/env python3
"""
Complete test runner for SIRV Integration Pipeline.
Generates synthetic data, runs integration, simulates FLAMES output, and generates evaluation reports.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import argparse

# Set paths to tools
os.environ["PATH"] = "/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/minimap2/2.26/bin:/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/SAMtools/1.21/bin:" + os.environ.get("PATH", "")

def setup_environment():
    """Set up the environment and path."""
    # Make sure we're in the right directory
    pipeline_dir = Path("/data/gpfs/projects/punim2251/sirv-integration-pipeline")
    os.chdir(pipeline_dir)
    
    # Set up Python path to find the pipeline module
    sys.path.insert(0, str(pipeline_dir))

def create_test_directories():
    """Create required directories for testing."""
    test_dir = Path("./test_data")
    output_dir = test_dir / "output"
    flames_dir = test_dir / "flames_output"
    eval_dir = test_dir / "evaluation"
    plots_dir = eval_dir / "plots"
    
    for dir_path in [test_dir, output_dir, flames_dir, eval_dir, plots_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return test_dir, output_dir, flames_dir, eval_dir, plots_dir

def run_integration_pipeline(coverage_model="10x_cdna", learn_from=None, visualize=True, seed=42):
    """Run the integration pipeline with specified coverage model."""
    print(f"Running integration pipeline with {coverage_model} coverage model...")
    
    # Create test directories
    test_dir, output_dir, flames_dir, eval_dir, plots_dir = create_test_directories()
    
    # Import the pipeline modules
    from sirv_pipeline.coverage_bias import CoverageBiasModel
    from sirv_pipeline.integration import add_sirv_to_dataset
    from sirv_pipeline.mapping import map_sirv_reads, create_alignment
    
    # Define file paths
    sirv_fastq = test_dir / "sirv_test.fastq"
    sc_fastq = test_dir / "sc_test.fastq"
    sirv_reference = test_dir / "sirv_reference.fa"
    sirv_gtf = test_dir / "sirv_annotation.gtf"
    
    # Generate synthetic SIRV FASTQ and annotation if they don't exist
    if not sirv_fastq.exists() or not sirv_reference.exists() or not sirv_gtf.exists():
        print("Generating synthetic SIRV data...")
        create_synthetic_sirv_data(sirv_fastq, sirv_reference, sirv_gtf)
    
    # Generate synthetic sc FASTQ if it doesn't exist
    if not sc_fastq.exists():
        print("Generating synthetic scRNA-seq data...")
        create_synthetic_sc_data(sc_fastq)
    
    # Map SIRV reads
    transcript_map_file = output_dir / "transcript_map.csv"
    alignment_file = output_dir / "sirv_alignment.bam"
    
    print("Mapping SIRV reads...")
    map_sirv_reads(
        sirv_fastq=str(sirv_fastq),
        sirv_reference=str(sirv_reference),
        sirv_gtf=str(sirv_gtf),
        output_csv=str(transcript_map_file),
        threads=4
    )
    
    create_alignment(
        fastq_file=str(sirv_fastq),
        reference_file=str(sirv_reference),
        output_bam=str(alignment_file),
        threads=4,
        preset="map-ont"
    )
    
    # Initialize coverage bias model
    coverage_model_file = output_dir / "coverage_model.json"
    
    if learn_from:
        # Learn bias from BAM file
        print(f"Learning coverage bias from {learn_from}...")
        coverage_model = CoverageBiasModel(model_type="custom", seed=seed)
        coverage_model.learn_from_bam(
            bam_file=learn_from,
            annotation_file=str(sirv_gtf),
            min_reads=10,
            length_bins=3
        )
    else:
        # Use default model
        print(f"Using default {coverage_model} coverage model...")
        coverage_model = CoverageBiasModel(model_type=coverage_model, seed=seed)
    
    # Save model
    coverage_model.save(str(coverage_model_file))
    
    # Visualize if requested
    if visualize:
        plot_file = output_dir / "coverage_bias.png"
        coverage_model.plot_distributions(str(plot_file))
    
    # Add SIRV reads to sc dataset
    integrated_fastq = output_dir / "integrated.fastq"
    tracking_file = output_dir / "tracking.csv"
    expected_file = output_dir / "expected_counts.csv"
    
    print("Adding SIRV reads to scRNA-seq dataset...")
    add_sirv_to_dataset(
        sc_fastq=str(sc_fastq),
        sirv_fastq=str(sirv_fastq),
        transcript_map_file=str(transcript_map_file),
        coverage_model_file=str(coverage_model_file),
        output_fastq=str(integrated_fastq),
        tracking_file=str(tracking_file),
        expected_file=str(expected_file),
        insertion_rate=0.05,
        coverage_model=coverage_model,
        seed=seed
    )
    
    print("Integration pipeline completed successfully")
    return tracking_file, expected_file

def create_synthetic_sirv_data(sirv_fastq, sirv_reference, sirv_gtf):
    """Create synthetic SIRV data for testing."""
    # Simple SIRV transcripts
    sirv_transcripts = {
        "SIRV1": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
        "SIRV2": "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA",
        "SIRV3": "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "SIRV4": "GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC",
        "SIRV5": "CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG"
    }
    
    # Write SIRV reference
    with open(sirv_reference, 'w') as f:
        for name, seq in sirv_transcripts.items():
            f.write(f">{name}\n{seq}\n")
    
    # Write SIRV GTF
    with open(sirv_gtf, 'w') as f:
        f.write('##gff-version 3\n')
        for name, seq in sirv_transcripts.items():
            length = len(seq)
            f.write(f"{name}\tSIRV\ttranscript\t1\t{length}\t.\t+\t.\ttranscript_id \"{name}\"; gene_id \"{name}\";\n")
            f.write(f"{name}\tSIRV\texon\t1\t{length}\t.\t+\t.\ttranscript_id \"{name}\"; gene_id \"{name}\";\n")
    
    # Write SIRV FASTQ
    with open(sirv_fastq, 'w') as f:
        read_id = 1
        for name, seq in sirv_transcripts.items():
            # Create multiple reads with different lengths for each transcript
            for i in range(10):
                start = np.random.randint(0, 20)
                length = np.random.randint(max(40, len(seq) - 30), len(seq))
                read_seq = seq[start:start+length]
                qual = 'I' * len(read_seq)  # Constant quality for simplicity
                
                f.write(f"@read_{read_id}_{name}\n")
                f.write(f"{read_seq}\n")
                f.write(f"+\n")
                f.write(f"{qual}\n")
                read_id += 1

def create_synthetic_sc_data(sc_fastq):
    """Create synthetic single-cell data for testing."""
    # Mock cell barcodes and UMIs
    cell_barcodes = ["AAACATGCTTGACTGG", "AAACGGGCACAGTCTA", "AAAGCGATCACCGTAT"]
    umis = ["ACGTACGTACGT", "TGCATGCATGCA", "GCTAGCTAGCTA", "GATCGATCGATC"]
    
    # Genes and sequences
    genes = {
        "GENE1": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        "GENE2": "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT",
        "GENE3": "TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG"
    }
    
    # Write scRNA-seq FASTQ
    with open(sc_fastq, 'w') as f:
        read_id = 1
        for gene_name, gene_seq in genes.items():
            for barcode in cell_barcodes:
                # Add multiple reads per gene/cell
                for i in range(np.random.randint(5, 15)):
                    umi = np.random.choice(umis)
                    
                    # Create reads with cell barcode + UMI + gene fragment
                    start = np.random.randint(0, 10)
                    length = np.random.randint(40, len(gene_seq) - start)
                    read_seq = barcode + umi + gene_seq[start:start+length]
                    qual = 'I' * len(read_seq)  # Constant quality for simplicity
                    
                    f.write(f"@read_{read_id}_{gene_name}_{barcode}\n")
                    f.write(f"{read_seq}\n")
                    f.write(f"+\n")
                    f.write(f"{qual}\n")
                    read_id += 1

def run_legacy_pipeline():
    """Run the existing test pipeline to generate integrated FASTQ."""
    try:
        subprocess.run(["python", "run_test_pipeline.py"], check=True)
        print("Integration pipeline completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Integration pipeline failed: {e}")
        return False

def simulate_flames_output(tracking_file, flames_output_file):
    """Simulate FLAMES transcript detection and quantification output."""
    print("Simulating FLAMES transcript detection and quantification...")
    
    # Load the tracking data
    tracking = pd.read_csv(tracking_file)
    
    # Extract barcodes and SIRV transcripts
    barcodes = tracking['barcode'].unique()
    sirv_transcripts = tracking['sirv_transcript'].unique()
    
    # Calculate expected counts per transcript-barcode pair
    expected_counts = tracking.groupby(['barcode', 'sirv_transcript']).size().reset_index()
    expected_counts.columns = ['barcode', 'sirv_transcript', 'expected_count']
    
    # Add some biological genes for realism
    bio_genes = [f"GENE_{i}" for i in range(1, 101)]
    
    # Create a FLAMES-like output dataframe
    flames_data = []
    
    # Add SIRV transcripts with realistic noise
    for _, row in expected_counts.iterrows():
        # Add some noise and detection failures
        detection_prob = min(0.9, 0.2 + (0.6 * np.log1p(row['expected_count']) / np.log1p(10)))
        
        if np.random.rand() < detection_prob:
            # Detected transcript with some count variation
            noise_factor = np.random.normal(1.0, 0.3)
            count = max(1, int(row['expected_count'] * noise_factor))
            
            flames_data.append({
                'transcript_id': row['sirv_transcript'],
                'barcode': row['barcode'],
                'UMI_count': count
            })
    
    # Add random biological genes
    for barcode in barcodes:
        # Each cell expresses a random subset of bio genes
        for gene in np.random.choice(bio_genes, size=np.random.randint(30, 70), replace=False):
            count = np.random.negative_binomial(5, 0.5)
            if count > 0:
                flames_data.append({
                    'transcript_id': gene,
                    'barcode': barcode,
                    'UMI_count': count
                })
    
    # Create DataFrame and save
    flames_df = pd.DataFrame(flames_data)
    os.makedirs(os.path.dirname(flames_output_file), exist_ok=True)
    flames_df.to_csv(flames_output_file, index=False)
    
    # Also create expected counts file directly from tracking info
    expected_counts.to_csv(Path(tracking_file).parent / "expected_counts.csv", index=False)
    
    print(f"Generated simulated FLAMES output with {len(flames_df)} transcript-barcode pairs")
    print(f"SIRV transcripts: {len(sirv_transcripts)}")
    print(f"Cells: {len(barcodes)}")
    
    return flames_df, expected_counts

def generate_plots(comparison_df, plots_dir):
    """Generate evaluation plots similar to what the pipeline would create."""
    print("Generating evaluation plots...")
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Count correlation plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        comparison_df['expected_count'], 
        comparison_df['observed_count'],
        alpha=0.6
    )
    plt.xlabel('Expected SIRV Count')
    plt.ylabel('Observed SIRV Count')
    plt.title('Expected vs Observed SIRV Counts')
    plt.grid(alpha=0.3)
    
    # Add diagonal line
    max_val = max(
        comparison_df['expected_count'].max(), 
        comparison_df['observed_count'].max()
    )
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'count_correlation.png'), dpi=100)
    plt.close()
    
    # 2. Detection rate plot
    plt.figure(figsize=(8, 6))
    
    # Add jitter to expected counts for better visualization
    jitter = np.random.normal(0, 0.1, size=len(comparison_df))
    jittered_expected = comparison_df['expected_count'] + jitter
    
    plt.scatter(
        jittered_expected, 
        comparison_df['detected'].astype(int),
        alpha=0.5
    )
    
    plt.xlabel('Expected SIRV Count')
    plt.ylabel('Detected (1=Yes, 0=No)')
    plt.title('SIRV Detection by Expected Count')
    plt.grid(alpha=0.3)
    
    # Add trend line using moving average
    sorted_data = comparison_df.sort_values('expected_count')
    window_size = max(10, len(sorted_data) // 10)
    x_vals = []
    y_vals = []
    
    for i in range(0, len(sorted_data) - window_size, window_size // 2):
        subset = sorted_data.iloc[i:i+window_size]
        x_vals.append(subset['expected_count'].mean())
        y_vals.append(subset['detected'].mean())
    
    plt.plot(x_vals, y_vals, 'r-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'detection_rate.png'), dpi=100)
    plt.close()
    
    # 3. Per-transcript detection rates
    transcript_stats = comparison_df.groupby('sirv_transcript').agg({
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
    plt.savefig(os.path.join(plots_dir, 'transcript_detection.png'), dpi=100)
    plt.close()
    
    return transcript_stats

def create_html_report(comparison_df, transcript_stats, eval_dir, plots_dir, output_file):
    """Create an HTML report summarizing evaluation results."""
    print("Generating HTML report...")
    
    # Calculate summary statistics
    total_expected = len(comparison_df[comparison_df['expected_count'] > 0])
    total_detected = len(comparison_df[comparison_df['detected']])
    detection_rate = total_detected / total_expected if total_expected > 0 else 0
    
    # Calculate correlation
    expected_counts = comparison_df['expected_count'].values
    observed_counts = comparison_df['observed_count'].values
    correlation = np.corrcoef(expected_counts, observed_counts)[0, 1]
    
    # Since plots_dir is already a subfolder of eval_dir, no need to copy files
    plots_folder = "plots"
    
    # Create HTML content with relative paths to the plots
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SIRV Integration Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .summary {{ margin: 20px 0; }}
            .plots {{ display: flex; flex-wrap: wrap; }}
            .plot {{ margin: 10px; max-width: 45%; }}
            .plot img {{ max-width: 100%; height: auto; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>SIRV Integration Evaluation Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total expected transcripts: {total_expected}</p>
            <p>Total detected transcripts: {total_detected}</p>
            <p>Overall detection rate: {detection_rate:.2%}</p>
            <p>Correlation coefficient: {correlation:.4f}</p>
        </div>
        
        <div class="plots">
            <div class="plot">
                <h2>Expected vs Observed Counts</h2>
                <img src="{plots_folder}/count_correlation.png" alt="Count Correlation">
            </div>
            <div class="plot">
                <h2>Detection Rate by Expected Count</h2>
                <img src="{plots_folder}/detection_rate.png" alt="Detection Rate">
            </div>
            <div class="plot">
                <h2>Detection Rate by SIRV Transcript</h2>
                <img src="{plots_folder}/transcript_detection.png" alt="Transcript Detection">
            </div>
        </div>
        
        <div class="transcript-table">
            <h2>Transcript-level Statistics</h2>
            <table>
                <tr>
                    <th>Transcript</th>
                    <th>Expected Count</th>
                    <th>Observed Count</th>
                    <th>Detection Rate</th>
                </tr>
    """
    
    # Add transcript stats to the table
    for _, row in transcript_stats.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['sirv_transcript']}</td>
                    <td>{int(row['expected_count'])}</td>
                    <td>{int(row['observed_count'])}</td>
                    <td>{row['detected']:.2%}</td>
                </tr>
        """
    
    # Close the HTML
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_file}")
    return output_file

def create_comparison_data(expected_counts, flames_df):
    """Create comparison data between expected and observed counts."""
    # Extract SIRV counts from FLAMES output
    sirv_counts = flames_df[flames_df['transcript_id'].str.contains('SIRV', case=False, na=False)].copy()
    
    # Standardize column names
    sirv_counts = sirv_counts.rename(columns={
        'transcript_id': 'sirv_transcript',
        'UMI_count': 'observed_count'
    })
    
    # Merge expected and observed
    comparison = pd.merge(
        expected_counts[['barcode', 'sirv_transcript', 'expected_count']],
        sirv_counts[['barcode', 'sirv_transcript', 'observed_count']],
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
    
    return comparison

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SIRV Integration Pipeline Test Runner")
    
    parser.add_argument("--coverage-model", type=str, choices=["10x_cdna", "direct_rna", "custom"], 
                        default="10x_cdna", help="Coverage bias model to use")
    parser.add_argument("--learn-from", type=str, help="BAM file to learn coverage from")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def run_complete_pipeline():
    """Run the complete test pipeline from integration to evaluation."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up environment
    setup_environment()
    
    # Create test directories
    test_dir, output_dir, flames_dir, eval_dir, plots_dir = create_test_directories()
    
    # Run integration pipeline
    tracking_file, expected_file = run_integration_pipeline(
        coverage_model=args.coverage_model,
        learn_from=args.learn_from,
        visualize=not args.no_visualize,
        seed=args.seed
    )
    
    # Simulate FLAMES output
    flames_output_file = flames_dir / "flames_output.csv"
    flames_df, expected_counts = simulate_flames_output(tracking_file, flames_output_file)
    
    # Create comparison data
    comparison_df = create_comparison_data(expected_counts, flames_df)
    comparison_file = eval_dir / "comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    
    # Generate plots
    transcript_stats = generate_plots(comparison_df, plots_dir)
    
    # Create HTML report
    report_file = eval_dir / "report.html"
    create_html_report(comparison_df, transcript_stats, eval_dir, plots_dir, report_file)
    
    print(f"Complete test pipeline finished. Results in {eval_dir}")
    print(f"HTML report: {report_file}")

if __name__ == "__main__":
    run_complete_pipeline() 