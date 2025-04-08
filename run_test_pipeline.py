#!/usr/bin/env python3
"""
Test data generator and runner for SIRV Integration Pipeline.
Creates synthetic data matching the required database schema and runs the pipeline.
"""

import os
import subprocess
import shutil
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

# Set paths to tools
os.environ["PATH"] = "/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/minimap2/2.26/bin:/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/SAMtools/1.21/bin:" + os.environ.get("PATH", "")

def create_directories() -> Tuple[Path, Path]:
    """Create test directories."""
    test_dir = Path("./test_data")
    output_dir = test_dir / "output"
    
    for dir_path in [test_dir, output_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return test_dir, output_dir

def generate_synthetic_data(test_dir: Path) -> Dict[str, Path]:
    """Generate synthetic SIRV and scRNA-seq data following the required schema."""
    print("Generating synthetic test data...")
    
    # Define paths for output files
    sirv_fastq = test_dir / "synthetic_sirv.fastq"
    sc_fastq = test_dir / "synthetic_scrna.fastq"
    sirv_reference = test_dir / "synthetic_sirv.fa"
    sirv_gtf = test_dir / "synthetic_sirv.gtf"
    transcript_map = test_dir / "transcript_map.csv"
    
    # Generate SIRV reference genome
    sirv_transcripts = {
        "SIRV101": 1500, "SIRV102": 1200, "SIRV103": 1000, "SIRV201": 2000,
        "SIRV202": 1800, "SIRV301": 1700, "SIRV302": 1300
    }
    
    # Create FASTA file
    with open(sirv_reference, 'w') as fasta_out:
        for transcript, length in sirv_transcripts.items():
            sequence = ''.join(np.random.choice(['A', 'C', 'G', 'T']) for _ in range(length))
            fasta_out.write(f">{transcript}\n{sequence}\n")
    
    # Create GTF file
    with open(sirv_gtf, 'w') as gtf_out:
        for transcript, length in sirv_transcripts.items():
            chrom = transcript[:5]
            gtf_out.write(f'{chrom}\tSIRV\ttranscript\t1\t{length}\t.\t+\t.\ttranscript_id "{transcript}"; gene_id "{chrom}";\n')
            gtf_out.write(f'{chrom}\tSIRV\texon\t1\t{length}\t.\t+\t.\ttranscript_id "{transcript}"; gene_id "{chrom}";\n')
    
    # Generate SIRV reads and transcript map
    create_sirv_reads_and_map(sirv_fastq, transcript_map, sirv_transcripts)
    
    # Generate scRNA-seq reads
    create_sc_reads(sc_fastq)
    
    print(f"Created SIRV reference FASTA: {sirv_reference}")
    print(f"Created SIRV reference GTF: {sirv_gtf}")
    print(f"Generated SIRV reads: {sirv_fastq}")
    print(f"Generated transcript map: {transcript_map}")
    print(f"Generated scRNA-seq reads: {sc_fastq}")
    
    return {
        "sirv_fastq": sirv_fastq,
        "sc_fastq": sc_fastq,
        "sirv_reference": sirv_reference,
        "sirv_gtf": sirv_gtf,
        "transcript_map": transcript_map
    }

def create_sirv_reads_and_map(fastq_path: Path, map_path: Path, sirv_transcripts: Dict[str, int]) -> None:
    """Create synthetic SIRV reads and corresponding transcript map following schema."""
    # Each transcript will have some reads
    reads_per_transcript = 30
    total_reads = sum(reads_per_transcript for _ in sirv_transcripts)
    
    # Create transcript mapping file
    # Schema: read_id, sirv_transcript, overlap_fraction, read_length, alignment_length
    with open(map_path, 'w') as map_file:
        map_file.write("read_id,sirv_transcript,overlap_fraction,read_length,alignment_length\n")
        
        with open(fastq_path, 'w') as fastq_file:
            read_id = 0
            for transcript, length in sirv_transcripts.items():
                for i in range(reads_per_transcript):
                    read_id += 1
                    read_name = f"sirv_read_{read_id}"
                    
                    # Generate a read with variable coverage of transcript
                    coverage = np.random.uniform(0.7, 0.99)
                    read_length = int(length * coverage)
                    
                    # Generate sequence
                    sequence = ''.join(np.random.choice(['A', 'C', 'G', 'T']) for _ in range(read_length))
                    
                    # Generate quality
                    quality = ''.join(chr(np.random.randint(33, 73)) for _ in range(read_length))
                    
                    # Write to FASTQ
                    fastq_file.write(f"@{read_name}\n{sequence}\n+\n{quality}\n")
                    
                    # Write to map file
                    map_file.write(f"{read_name},{transcript},{coverage:.2f},{read_length},{read_length}\n")
    
    print(f"Generated {total_reads} synthetic SIRV reads with transcript mapping")

def create_sc_reads(fastq_path: Path, num_reads: int = 500, num_cells: int = 5) -> None:
    """Create synthetic scRNA-seq reads with barcodes."""
    # Generate cell barcodes
    barcodes = [
        ''.join(np.random.choice(['A', 'C', 'G', 'T']) for _ in range(16))
        for _ in range(num_cells)
    ]
    
    with open(fastq_path, 'w') as f:
        for i in range(num_reads):
            # Select a random barcode
            barcode = np.random.choice(barcodes)
            
            # Generate UMI
            umi = ''.join(np.random.choice(['A', 'C', 'G', 'T']) for _ in range(10))
            
            # Create read ID with barcode and UMI
            read_id = f"scread_{i+1}_{barcode}_{umi}"
            
            # Generate variable length read
            length = np.random.randint(800, 1500)
            sequence = ''.join(np.random.choice(['A', 'C', 'G', 'T']) for _ in range(length))
            quality = ''.join(chr(np.random.randint(33, 73)) for _ in range(length))
            
            # Write to FASTQ
            f.write(f"@{read_id}\n{sequence}\n+\n{quality}\n")
    
    print(f"Generated {num_reads} synthetic scRNA-seq reads across {num_cells} cells")

def run_pipeline(data_files: Dict[str, Path], output_dir: Path) -> None:
    """Run the SIRV Integration Pipeline with synthetic data."""
    print("Running SIRV Integration Pipeline...")
    
    # Build the command to run the pipeline
    cmd = [
        "python", "-m", "sirv_pipeline",
        "--integration",
        "--sirv-fastq", str(data_files["sirv_fastq"]),
        "--sc-fastq", str(data_files["sc_fastq"]),
        "--sirv-reference", str(data_files["sirv_reference"]),
        "--sirv-gtf", str(data_files["sirv_gtf"]),
        "--output-dir", str(output_dir),
        "--insertion-rate", "0.05",
        "--verbose"
    ]
    
    # Copy transcript map to output directory
    shutil.copy(data_files["transcript_map"], output_dir / "transcript_map.csv")
    
    try:
        # Run the pipeline
        subprocess.run(cmd, check=True)
        print("Pipeline execution completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Pipeline execution failed: {e}")
        print("Trying direct module import approach...")
        
        try:
            # Import necessary modules
            from sirv_pipeline.integration import add_sirv_to_dataset
            from sirv_pipeline.coverage_bias import create_coverage_bias_model
            
            # Setup output files
            integrated_fastq = output_dir / "integrated.fastq"
            tracking_file = output_dir / "tracking.csv"
            coverage_model_file = output_dir / "coverage_model.csv"
            
            # Create coverage model
            length_sampler, bias_model = create_coverage_bias_model(
                fastq_file=str(data_files["sc_fastq"]),
                reference_file=str(data_files["sirv_reference"]),
                sample_size=100
            )
            
            # Directly call integration function
            add_sirv_to_dataset(
                sc_fastq=str(data_files["sc_fastq"]),
                sirv_fastq=str(data_files["sirv_fastq"]),
                transcript_map_file=str(data_files["transcript_map"]),
                coverage_model_file=str(coverage_model_file),
                output_fastq=str(integrated_fastq),
                tracking_file=str(tracking_file),
                insertion_rate=0.05
            )
            
            print("Direct module integration completed successfully")
        except Exception as e2:
            print(f"Direct approach also failed: {e2}")

def verify_outputs(output_dir: Path) -> None:
    """Verify output files match the required database schema."""
    tracking_file = output_dir / "tracking.csv"
    
    if tracking_file.exists():
        try:
            df = pd.read_csv(tracking_file)
            required_columns = [
                "read_id", "original_read_id", "barcode", "umi", 
                "sirv_transcript", "original_length", "modified_length"
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Output is missing columns: {missing_columns}")
            else:
                print("Output schema verification passed")
                
            print(f"Tracking info: {len(df)} reads, {df['sirv_transcript'].nunique()} transcripts, {df['barcode'].nunique()} cells")
        except Exception as e:
            print(f"Error verifying outputs: {e}")
    else:
        print(f"Warning: Expected output file not found: {tracking_file}")

def main():
    """Main function to generate test data and run the pipeline."""
    # Make sure we're in the right directory
    pipeline_dir = Path("/data/gpfs/projects/punim2251/sirv-integration-pipeline")
    os.chdir(pipeline_dir)
    
    # Set up Python path to find the pipeline module
    sys.path.insert(0, str(pipeline_dir))
    
    # Create test directories
    test_dir, output_dir = create_directories()
    
    # Generate synthetic data
    data_files = generate_synthetic_data(test_dir)
    
    # Run the pipeline
    run_pipeline(data_files, output_dir)
    
    # Verify outputs
    verify_outputs(output_dir)
    
    # Print output file locations
    print("\nTest data generation and pipeline run completed!")
    print("\nOutput files:")
    for file_path in output_dir.glob("*"):
        print(f"- {file_path}")

if __name__ == "__main__":
    main() 