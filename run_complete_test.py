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

def run_integration_pipeline():
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

def run_complete_pipeline():
    """Run the complete pipeline from integration to evaluation."""
    # Set up environment
    setup_environment()
    
    # Create directories
    test_dir, output_dir, flames_dir, eval_dir, plots_dir = create_test_directories()
    
    # Run integration pipeline using direct module approach instead of subprocess
    print("Running integration pipeline using direct module approach...")
    
    try:
        # First check if run_test_pipeline.py exists and run it
        if Path("run_test_pipeline.py").exists():
            import run_test_pipeline
            run_test_pipeline.main()
            print("Integration pipeline completed successfully")
        else:
            print("run_test_pipeline.py not found, skipping test data generation")
            print("Make sure test data is already available in test_data directory")
    except Exception as e:
        print(f"Error running integration pipeline: {e}")
        print("Check that test data has been properly generated")
        return
    
    # Paths for files
    tracking_file = output_dir / "tracking.csv"
    coverage_model_file = output_dir / "coverage_model.csv"
    flames_output_file = flames_dir / "simulated_flames_output.csv"
    comparison_file = eval_dir / "comparison.csv"
    report_file = eval_dir / "report.html"
    
    # Verify tracking file exists
    if not tracking_file.exists():
        print(f"Error: Tracking file not found at {tracking_file}")
        print("Integration pipeline did not complete successfully")
        return
    
    # Simulate FLAMES output
    flames_df, expected_counts = simulate_flames_output(tracking_file, flames_output_file)
    
    # Create comparison data
    comparison_df = create_comparison_data(expected_counts, flames_df)
    
    # Save comparison data
    comparison_df.to_csv(comparison_file, index=False)
    
    # Generate plots
    transcript_stats = generate_plots(comparison_df, plots_dir)
    
    # Create HTML report
    create_html_report(comparison_df, transcript_stats, eval_dir, plots_dir, report_file)
    
    # Create a simple HTML index to open the report
    index_html = eval_dir / "index.html"
    with open(index_html, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SIRV Evaluation Results</title>
            <meta http-equiv="refresh" content="0; url='report.html'" />
        </head>
        <body>
            <p>Redirecting to <a href="report.html">report.html</a>...</p>
        </body>
        </html>
        """)
    
    # Generate additional diagnostic visualizations
    try:
        print("\nGenerating additional diagnostic visualizations...")
        try:
            import seaborn
            print("Seaborn is available - full diagnostics will be generated")
        except ImportError:
            print("Warning: Seaborn is not installed - limited diagnostics will be available")
            print("To enable full diagnostics, install seaborn: pip install seaborn")
        
        from sirv_pipeline.diagnostics import create_diagnostic_visualizations
        
        diagnostic_plots = create_diagnostic_visualizations(
            tracking_file=tracking_file,
            coverage_model_file=coverage_model_file,
            eval_dir=eval_dir
        )
        print(f"Diagnostic visualizations created in {eval_dir}/diagnostics/")
    except Exception as e:
        print(f"Warning: Could not generate diagnostic visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nComplete pipeline run finished!")
    print("\nOutput files:")
    print(f"- Integration output: {output_dir}")
    print(f"- Simulated FLAMES output: {flames_output_file}")
    print(f"- Evaluation results: {eval_dir}")
    print(f"- Visualization plots: {plots_dir}")
    print(f"- HTML report: {report_file}")
    print(f"- HTML index: {index_html}")
    print(f"- Diagnostic visualizations: {eval_dir}/diagnostics/diagnostics.html")
    print("\nTo view the reports, open the html files in a web browser.")

if __name__ == "__main__":
    run_complete_pipeline() 