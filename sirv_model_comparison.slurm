#!/bin/bash
#SBATCH --partition="sapphire"
#SBATCH --nodes=1
#SBATCH --account="punim2251"
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mschauhan@student.unimelb.edu.au
#SBATCH --cpus-per-task=12
#SBATCH --mem=200gb
#SBATCH --time=0-12:00:00
#SBATCH --job-name="sirv_model_compare"

# Create timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Define pipeline directory
PIPELINE_DIR="/data/gpfs/projects/punim2251/sirv-integration-pipeline"

# Define input files and parameters
SIRV_BAM="/data/gpfs/projects/punim0646/ric/PMbrain_cDNAPCR/sirv_bams/bc01_sirv_reads.bam"
SIRV_REF="/data/gpfs/projects/punim0646/sirv_genomes/sirv_transcriptome_c.fa"
SC_FASTQ="/data/gpfs/projects/punim2251/Aim1_LongBench/ReadRarefaction_wFixedCells/data/LongBench_All/ont_sc/10Percent_FLAMES/matched_reads.fastq"
FLAMES_BAM="/data/gpfs/projects/punim2251/Aim1_LongBench/ReadRarefaction_wFixedCells/data/LongBench_All/ont_sc/10Percent_FLAMES/realign2transcript.bam"
FLAMES_GTF="/data/gpfs/projects/punim2251/Aim1_LongBench/ReadRarefaction_wFixedCells/data/LongBench_All/ont_sc/10Percent_FLAMES/isoform_annotated.gtf"

# Create new output directories
BASE_OUTPUT_DIR="/data/gpfs/projects/punim2251/sirv_run_${TIMESTAMP}"
RF_OUTPUT_DIR="${BASE_OUTPUT_DIR}/random_forest_results"
GB_OUTPUT_DIR="${BASE_OUTPUT_DIR}/gradient_boosting_results"
COMPARISON_DIR="${BASE_OUTPUT_DIR}/model_comparison"

echo "Creating output directories:"
echo "  Base: ${BASE_OUTPUT_DIR}"
echo "  Random Forest: ${RF_OUTPUT_DIR}"
echo "  Gradient Boosting: ${GB_OUTPUT_DIR}"
echo "  Comparison: ${COMPARISON_DIR}"

# Create output directories
mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${RF_OUTPUT_DIR}
mkdir -p ${GB_OUTPUT_DIR}
mkdir -p ${COMPARISON_DIR}
mkdir -p ${COMPARISON_DIR}/plots

# Copy important configuration files
INSERTION_RATE="0.01"
THREADS="12"  # Set to match cpus-per-task

# Load required modules
module load GCCcore/11.3.0
module load minimap2/2.26
module load GCC/11.3.0
module load SAMtools/1.21
module load Python/3.10.4

# Check if all input files exist
declare -a INPUT_FILES=(
    "${SIRV_BAM}"
    "${SIRV_REF}"
    "${SC_FASTQ}"
    "${FLAMES_BAM}"
    "${FLAMES_GTF}"
)

for file in "${INPUT_FILES[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "ERROR: Input file not found at ${file}"
        echo "Please check the path and try again."
        exit 1
    fi
done

# Create and index BAM files properly
echo "Creating proper BAM files with complete indices..."

# Create a fixed copy of the FLAMES BAM with proper index
FIXED_FLAMES_BAM="${BASE_OUTPUT_DIR}/fixed_flames.bam"
echo "Creating fixed FLAMES BAM at ${FIXED_FLAMES_BAM}..."
samtools view -h ${FLAMES_BAM} | samtools sort -o ${FIXED_FLAMES_BAM}
samtools index ${FIXED_FLAMES_BAM}

# Create a fixed copy of the SIRV BAM with proper index
FIXED_SIRV_BAM="${BASE_OUTPUT_DIR}/fixed_sirv.bam"
echo "Creating fixed SIRV BAM at ${FIXED_SIRV_BAM}..."
samtools view -h ${SIRV_BAM} | samtools sort -o ${FIXED_SIRV_BAM}
samtools index ${FIXED_SIRV_BAM}

# Change to pipeline directory and activate virtual environment
cd ${PIPELINE_DIR}
source sirv_env/bin/activate

# Install/update the package and required dependencies for both models
echo "Installing/updating the SIRV integration pipeline package and ML dependencies..."
pip install -e .
pip install seaborn scikit-learn>=1.0.0 pyfaidx biopython matplotlib>=3.5.0 pandas>=1.0.0

# Function to run a single model
run_model() {
    model_type=$1
    output_dir=$2
    model_start_time=$SECONDS
    
    echo "Running pipeline with ${model_type} model..."
    
    # Run the pipeline - Fixed: replaced --flames-gtf with --annotation-file
    python -m sirv_pipeline --integration \
        --sirv-bam ${FIXED_SIRV_BAM} \
        --sirv-reference ${SIRV_REF} \
        --sc-fastq ${SC_FASTQ} \
        --output-dir ${output_dir} \
        --insertion-rate ${INSERTION_RATE} \
        --threads ${THREADS} \
        --coverage-model ${model_type} \
        --learn-coverage-from ${FIXED_FLAMES_BAM} \
        --annotation-file ${FLAMES_GTF} \
        --min-reads-for-learning 5 \
        --length-bins 5 \
        --extract-features \
        --visualize-coverage \
        --run-comparative-analysis \
        --verbose
    
    model_runtime=$((SECONDS - model_start_time))
    echo "Pipeline completed with ${model_type} model in ${model_runtime} seconds"
    echo "${model_type} model runtime: ${model_runtime} seconds" > "${output_dir}/model_timing.txt"
    
    # Save copy of model and key files
    if [ -f "${output_dir}/coverage_model.pkl" ]; then
        cp ${output_dir}/coverage_model.pkl ${COMPARISON_DIR}/${model_type}_model.pkl
    else
        echo "Warning: ${output_dir}/coverage_model.pkl not found"
    fi
    
    if [ -f "${output_dir}/improved_coverage_bias.png" ]; then
        cp ${output_dir}/improved_coverage_bias.png ${COMPARISON_DIR}/${model_type}_coverage_bias.png
    else
        echo "Warning: ${output_dir}/improved_coverage_bias.png not found"
    fi
    
    # Extract performance metrics from log file
    echo "Extracting performance metrics for ${model_type}..."
    if [ -f "${output_dir}/pipeline.log" ]; then
        grep -A 5 "Model performance:" ${output_dir}/pipeline.log > ${COMPARISON_DIR}/${model_type}_performance.txt
    else
        echo "Warning: ${output_dir}/pipeline.log not found"
        echo "No metrics available" > ${COMPARISON_DIR}/${model_type}_performance.txt
    fi
    
    # Extract coverage comparison metrics from comparative_bamslam
    if [ -d "${output_dir}/comparative_bamslam" ]; then
        if [ -d "${output_dir}/comparative_bamslam/plots" ]; then
            mkdir -p ${COMPARISON_DIR}/plots/${model_type}
            cp -r ${output_dir}/comparative_bamslam/plots/* ${COMPARISON_DIR}/plots/${model_type}/
        fi
    fi
}

# Run Random Forest model first
echo "Starting Random Forest model evaluation..."
run_model "random_forest" ${RF_OUTPUT_DIR}

# Run Gradient Boosting model second
echo "Starting Gradient Boosting model evaluation..."
run_model "ml_gradient_boosting" ${GB_OUTPUT_DIR}

# Create a comparison report
echo "Creating comparison report..."
cat > ${COMPARISON_DIR}/generate_comparison.py << 'EOL'
#!/usr/bin/env python
"""
Script to generate a comparison report between Random Forest and Gradient Boosting models
"""
import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Get the directory this script is running in
script_dir = os.path.dirname(os.path.abspath(__file__))
comparison_dir = script_dir
rf_perf_file = os.path.join(comparison_dir, "random_forest_performance.txt")
gb_perf_file = os.path.join(comparison_dir, "ml_gradient_boosting_performance.txt")
output_report = os.path.join(comparison_dir, "model_comparison_report.md")
output_csv = os.path.join(comparison_dir, "model_comparison_metrics.csv")

# Function to extract metrics from performance file
def extract_metrics(perf_file):
    metrics = {}
    try:
        with open(perf_file, 'r') as f:
            content = f.read()
            print(f"Content from {perf_file}:\n{content[:500]}...")
            
            # Extract MSE and R² values
            mse_match = re.search(r'MSE=(\d+\.\d+)', content)
            r2_match = re.search(r'R²=(\d+\.\d+)', content)
            
            if mse_match:
                metrics['MSE'] = float(mse_match.group(1))
                print(f"Found MSE: {metrics['MSE']}")
            else:
                print(f"No MSE match found in content")
                
            if r2_match:
                metrics['R2'] = float(r2_match.group(1))
                print(f"Found R2: {metrics['R2']}")
            else:
                print(f"No R2 match found in content")
            
            # Get model type from filename
            model_type = os.path.basename(perf_file).split('_')[0]
            
            # Initialize timing_file before using it
            timing_file = None
            
            # For random forest
            if "random_forest" in model_type:
                timing_file = os.path.join(comparison_dir, "../random_forest_results/model_timing.txt")
            # For gradient boosting
            elif "ml" in model_type or "gradient" in model_type:
                timing_file = os.path.join(comparison_dir, "../gradient_boosting_results/model_timing.txt")
            
            if timing_file and os.path.exists(timing_file):
                with open(timing_file, 'r') as tf:
                    timing_content = tf.read()
                    print(f"Timing content from {timing_file}:\n{timing_content}")
                    time_match = re.search(r'runtime: (\d+) seconds', timing_content)
                    if time_match:
                        metrics['Runtime'] = int(time_match.group(1))
                        print(f"Found Runtime: {metrics['Runtime']}")
            else:
                print(f"Timing file not found or not defined: {timing_file}")
            
            return metrics
    except Exception as e:
        print(f"Error extracting metrics from {perf_file}: {e}")
        import traceback
        print(traceback.format_exc())
        return {}

# Extract metrics for both models
print("Extracting Random Forest metrics...")
rf_metrics = extract_metrics(rf_perf_file)
print(f"RF metrics: {rf_metrics}")

print("Extracting Gradient Boosting metrics...")
gb_metrics = extract_metrics(gb_perf_file)
print(f"GB metrics: {gb_metrics}")

# Create comparison dataframe
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'R2', 'Runtime (s)'],
    'Random Forest': [rf_metrics.get('MSE', 'N/A'), rf_metrics.get('R2', 'N/A'), rf_metrics.get('Runtime', 'N/A')],
    'Gradient Boosting': [gb_metrics.get('MSE', 'N/A'), gb_metrics.get('R2', 'N/A'), gb_metrics.get('Runtime', 'N/A')]
})

# Save metrics to CSV
metrics_df.to_csv(output_csv, index=False)
print(f"Saved metrics to CSV: {output_csv}")

# Create comparison plots
plt.figure(figsize=(12, 10))

# Plot MSE comparison
plt.subplot(2, 2, 1)
mse_values = [rf_metrics.get('MSE', 0), gb_metrics.get('MSE', 0)]
plt.bar(['Random Forest', 'Gradient Boosting'], mse_values, color=['#3498db', '#e74c3c'])
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE Comparison (lower is better)')
for i, v in enumerate(mse_values):
    if isinstance(v, (int, float)):
        plt.text(i, v + 0.0001, f"{v:.4f}", ha='center')

# Plot R² comparison
plt.subplot(2, 2, 2)
r2_values = [rf_metrics.get('R2', 0), gb_metrics.get('R2', 0)]
plt.bar(['Random Forest', 'Gradient Boosting'], r2_values, color=['#3498db', '#e74c3c'])
plt.ylabel('R² Score')
plt.title('R² Comparison (higher is better)')
for i, v in enumerate(r2_values):
    if isinstance(v, (int, float)):
        plt.text(i, v - 0.02, f"{v:.4f}", ha='center')

# Plot runtime comparison
plt.subplot(2, 2, 3)
runtime_values = [rf_metrics.get('Runtime', 0), gb_metrics.get('Runtime', 0)]
plt.bar(['Random Forest', 'Gradient Boosting'], runtime_values, color=['#3498db', '#e74c3c'])
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Comparison (lower is better)')
for i, v in enumerate(runtime_values):
    if isinstance(v, (int, float)):
        plt.text(i, v + 10, f"{v}", ha='center')

# Add a table with all metrics
plt.subplot(2, 2, 4)
plt.axis('off')
table_data = [
    ['Metric', 'Random Forest', 'Gradient Boosting', 'Difference', 'Winner'],
    ['MSE', rf_metrics.get('MSE', 'N/A'), gb_metrics.get('MSE', 'N/A'), 
     f"{abs(rf_metrics.get('MSE', 0) - gb_metrics.get('MSE', 0)):.4f}" if isinstance(rf_metrics.get('MSE'), (int, float)) and isinstance(gb_metrics.get('MSE'), (int, float)) else 'N/A',
     'RF' if isinstance(rf_metrics.get('MSE'), (int, float)) and isinstance(gb_metrics.get('MSE'), (int, float)) and rf_metrics.get('MSE', float('inf')) < gb_metrics.get('MSE', float('inf')) else 'GB'],
    ['R²', rf_metrics.get('R2', 'N/A'), gb_metrics.get('R2', 'N/A'), 
     f"{abs(rf_metrics.get('R2', 0) - gb_metrics.get('R2', 0)):.4f}" if isinstance(rf_metrics.get('R2'), (int, float)) and isinstance(gb_metrics.get('R2'), (int, float)) else 'N/A',
     'GB' if isinstance(rf_metrics.get('R2'), (int, float)) and isinstance(gb_metrics.get('R2'), (int, float)) and gb_metrics.get('R2', 0) > rf_metrics.get('R2', 0) else 'RF'],
    ['Runtime (s)', rf_metrics.get('Runtime', 'N/A'), gb_metrics.get('Runtime', 'N/A'),
     f"{abs(rf_metrics.get('Runtime', 0) - gb_metrics.get('Runtime', 0))}" if isinstance(rf_metrics.get('Runtime'), (int, float)) and isinstance(gb_metrics.get('Runtime'), (int, float)) else 'N/A',
     'RF' if isinstance(rf_metrics.get('Runtime'), (int, float)) and isinstance(gb_metrics.get('Runtime'), (int, float)) and rf_metrics.get('Runtime', float('inf')) < gb_metrics.get('Runtime', float('inf')) else 'GB']
]

plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
plt.title('Performance Comparison Summary', pad=20)

plt.suptitle('Coverage Model Comparison: Random Forest vs Gradient Boosting', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(comparison_dir, 'model_comparison_metrics.png'), dpi=300)
print(f"Saved comparison plot to: {os.path.join(comparison_dir, 'model_comparison_metrics.png')}")

# Generate a markdown report
with open(output_report, 'w') as f:
    f.write("# Coverage Model Comparison Report\n\n")
    f.write("## Performance Metrics\n\n")
    f.write("| Metric | Random Forest | Gradient Boosting | Difference | Better Model |\n")
    f.write("|--------|--------------|-------------------|------------|--------------|\n")
    
    # MSE (lower is better)
    rf_mse = rf_metrics.get('MSE', 'N/A')
    gb_mse = gb_metrics.get('MSE', 'N/A')
    if isinstance(rf_mse, (int, float)) and isinstance(gb_mse, (int, float)):
        mse_diff = abs(rf_mse - gb_mse)
        mse_better = 'Random Forest' if rf_mse < gb_mse else 'Gradient Boosting'
        f.write(f"| MSE | {rf_mse:.6f} | {gb_mse:.6f} | {mse_diff:.6f} | {mse_better} |\n")
    else:
        f.write(f"| MSE | {rf_mse} | {gb_mse} | N/A | N/A |\n")
    
    # R² (higher is better)
    rf_r2 = rf_metrics.get('R2', 'N/A')
    gb_r2 = gb_metrics.get('R2', 'N/A')
    if isinstance(rf_r2, (int, float)) and isinstance(gb_r2, (int, float)):
        r2_diff = abs(rf_r2 - gb_r2)
        r2_better = 'Gradient Boosting' if gb_r2 > rf_r2 else 'Random Forest'
        f.write(f"| R² | {rf_r2:.6f} | {gb_r2:.6f} | {r2_diff:.6f} | {r2_better} |\n")
    else:
        f.write(f"| R² | {rf_r2} | {gb_r2} | N/A | N/A |\n")
    
    # Runtime (lower is better)
    rf_runtime = rf_metrics.get('Runtime', 'N/A')
    gb_runtime = gb_metrics.get('Runtime', 'N/A')
    if isinstance(rf_runtime, (int, float)) and isinstance(gb_runtime, (int, float)):
        runtime_diff = abs(rf_runtime - gb_runtime)
        runtime_better = 'Random Forest' if rf_runtime < gb_runtime else 'Gradient Boosting'
        f.write(f"| Runtime (s) | {rf_runtime} | {gb_runtime} | {runtime_diff} | {runtime_better} |\n")
    else:
        f.write(f"| Runtime (s) | {rf_runtime} | {gb_runtime} | N/A | N/A |\n")
    
    # Overall recommendation
    f.write("\n## Overall Recommendation\n\n")
    
    if isinstance(rf_r2, (int, float)) and isinstance(gb_r2, (int, float)):
        if gb_r2 > rf_r2 and (gb_r2 - rf_r2) > 0.01:
            f.write("**Recommendation:** Use the **Gradient Boosting** model for better accuracy (R² score).\n")
        elif rf_r2 > gb_r2 and (rf_r2 - gb_r2) > 0.01:
            f.write("**Recommendation:** Use the **Random Forest** model for better accuracy (R² score).\n")
        else:
            # Models are similar in accuracy
            if isinstance(rf_runtime, (int, float)) and isinstance(gb_runtime, (int, float)):
                if rf_runtime < gb_runtime:
                    f.write("**Recommendation:** Both models have similar accuracy, but **Random Forest** is faster.\n")
                else:
                    f.write("**Recommendation:** Both models have similar accuracy, but **Gradient Boosting** is faster.\n")
            else:
                f.write("**Recommendation:** Both models have similar accuracy. Choose based on other factors.\n")
    else:
        f.write("**Recommendation:** Incomplete metrics - cannot make a definitive recommendation.\n")
    
    # Image comparisons
    f.write("\n## Visual Comparisons\n\n")
    f.write("### Performance Metrics\n\n")
    f.write("![Model Performance Comparison](model_comparison_metrics.png)\n\n")
    
    # Check if coverage bias plots exist before including them
    rf_bias_plot = os.path.join(comparison_dir, "random_forest_coverage_bias.png")
    gb_bias_plot = os.path.join(comparison_dir, "ml_gradient_boosting_coverage_bias.png")
    
    if os.path.exists(rf_bias_plot) or os.path.exists(gb_bias_plot):
        f.write("### Coverage Bias Plots\n\n")
        
        if os.path.exists(rf_bias_plot):
            f.write("#### Random Forest\n\n")
            f.write("![Random Forest Coverage Bias](random_forest_coverage_bias.png)\n\n")
        
        if os.path.exists(gb_bias_plot):
            f.write("#### Gradient Boosting\n\n")
            f.write("![Gradient Boosting Coverage Bias](ml_gradient_boosting_coverage_bias.png)\n\n")

    # Comparative Plots
    rf_plot_dir = os.path.join(comparison_dir, "plots/random_forest")
    gb_plot_dir = os.path.join(comparison_dir, "plots/ml_gradient_boosting")
    
    if os.path.isdir(rf_plot_dir) or os.path.isdir(gb_plot_dir):
        f.write("\n## Comparative Coverage Analysis\n\n")
        
        # Random Forest comparative plots
        if os.path.isdir(rf_plot_dir):
            plot_files = [f for f in os.listdir(rf_plot_dir) if f.endswith('.png')]
            if plot_files:
                f.write("### Random Forest Model\n\n")
                for plot in plot_files:
                    plot_path = f"plots/random_forest/{plot}"
                    plot_title = plot.replace('_', ' ').replace('.png', '')
                    f.write(f"#### {plot_title}\n\n")
                    f.write(f"![{plot_title}]({plot_path})\n\n")
        
        # Gradient Boosting comparative plots
        if os.path.isdir(gb_plot_dir):
            plot_files = [f for f in os.listdir(gb_plot_dir) if f.endswith('.png')]
            if plot_files:
                f.write("### Gradient Boosting Model\n\n")
                for plot in plot_files:
                    plot_path = f"plots/ml_gradient_boosting/{plot}"
                    plot_title = plot.replace('_', ' ').replace('.png', '')
                    f.write(f"#### {plot_title}\n\n")
                    f.write(f"![{plot_title}]({plot_path})\n\n")

print(f"Comparison report generated at {output_report}")
EOL

# Make the script executable
chmod +x ${COMPARISON_DIR}/generate_comparison.py

# Run the comparison script
cd ${COMPARISON_DIR}
python generate_comparison.py

# Create a README file
cat > ${BASE_OUTPUT_DIR}/README.md << EOF
# SIRV Integration Pipeline Model Comparison

This directory contains a comprehensive comparison between Random Forest and Gradient Boosting models
for the SIRV Integration Pipeline. The comparison was run on ${TIMESTAMP}.

## Directory Structure

- \`random_forest_results/\`: Results from running the pipeline with Random Forest model
- \`gradient_boosting_results/\`: Results from running the pipeline with Gradient Boosting model
- \`model_comparison/\`: Comparison of both models, including metrics and visualizations

## Key Files

- \`model_comparison/model_comparison_report.md\`: Comprehensive report comparing both models
- \`model_comparison/model_comparison_metrics.png\`: Visual comparison of performance metrics
- \`model_comparison/model_comparison_metrics.csv\`: CSV file with raw metrics for both models

## How to View Results

To view the comprehensive comparison report, open:
\`\`\`
${BASE_OUTPUT_DIR}/model_comparison/model_comparison_report.md
\`\`\`

## Input Datasets

- SIRV BAM: ${SIRV_BAM}
- SIRV Reference: ${SIRV_REF}
- scRNA-seq FASTQ: ${SC_FASTQ}
- FLAMES BAM (for coverage modeling): ${FLAMES_BAM}
- FLAMES GTF: ${FLAMES_GTF}

## Pipeline Parameters

- Insertion Rate: ${INSERTION_RATE}
- Threads: ${THREADS}
- Min Reads for Learning: 5
- Length Bins: 5
EOF

# Copy the script to the output directory for reproducibility
cp $0 ${BASE_OUTPUT_DIR}/

echo "Model comparison setup complete!"
echo "Results will be available in: ${BASE_OUTPUT_DIR}"
echo "Comprehensive report will be at: ${COMPARISON_DIR}/model_comparison_report.md"

# Deactivate virtual environment
deactivate 