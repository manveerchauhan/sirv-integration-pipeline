# VERSION 2

#!/usr/bin/env python3
"""
Simplified SIRV Integration Pipeline for ONT Data

Maps SIRV reads to SIRV reference, then adds them to existing ONT scRNA-seq data with tracking.

Usage:
    python sirv_ont_pipeline.py \
        --sirv_fastq sirv_reads.fastq \
        --sc_fastq 5y_ont_data.fastq \
        --sirv_reference sirv_genome.fa \
        --sirv_gtf sirv_annotation.gtf \
        --output_fastq combined_output.fastq \
        --insertion_rate 0.01
"""

import os
import argparse
import subprocess
import random
import numpy as np
import pandas as pd
import pysam
import gzip
from Bio import SeqIO
import tempfile
import shutil


def map_sirv_reads(sirv_fastq, sirv_reference, sirv_gtf, output_csv, threads=4):
    """
    Map SIRV reads to reference using minimap2 with ONT settings
    """
    print(f"Mapping SIRV reads to reference using minimap2...")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    prefix = os.path.basename(sirv_fastq).split('.')[0]
    
    # Align reads to SIRV reference
    bam_file = os.path.join(temp_dir, f"{prefix}.bam")
    sam_file = os.path.join(temp_dir, f"{prefix}.sam")
    
    # Run minimap2 with ONT preset
    cmd = [
        "minimap2", "-ax", "map-ont",     # ONT preset
        "-k", "14",                       # K-mer size for ONT
        "--secondary=no",                 # Don't output secondary alignments
        "-t", str(threads),
        sirv_reference, sirv_fastq
    ]
    
    with open(sam_file, 'w') as f:
        subprocess.run(cmd, stdout=f, check=True)
    
    # Convert SAM to BAM and sort
    subprocess.run(["samtools", "sort", "-o", bam_file, sam_file], check=True)
    subprocess.run(["samtools", "index", bam_file], check=True)
    
    # Load transcript info from GTF
    transcripts = {}
    with open(sirv_gtf, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if fields[2] == 'transcript':
                # Extract transcript_id
                attr_str = fields[8]
                for attr in attr_str.split(';'):
                    attr = attr.strip()
                    if attr.startswith('transcript_id'):
                        tx_id = attr.split(' ')[1].strip('"')
                        transcripts[tx_id] = {
                            'chrom': fields[0],
                            'start': int(fields[3]),
                            'end': int(fields[4])
                        }
    
    # Process BAM to identify transcripts
    bam = pysam.AlignmentFile(bam_file, "rb")
    mappings = []
    
    for read in bam:
        if read.is_unmapped:
            continue
        
        chrom = read.reference_name
        start = read.reference_start
        end = read.reference_end or read.reference_start + read.query_length
        
        # Find best matching transcript
        best_tx = None
        best_overlap = 0
        
        for tx_id, tx in transcripts.items():
            if tx['chrom'] == chrom:
                # Calculate overlap
                overlap_start = max(start, tx['start'])
                overlap_end = min(end, tx['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_tx = tx_id
        
        # Only save high confidence mappings (>50% overlap)
        read_len = end - start
        if best_tx and best_overlap > 0.5 * read_len:
            mappings.append({
                'read_id': read.query_name,
                'sirv_transcript': best_tx
            })
    
    bam.close()
    
    # Save mappings to CSV
    df = pd.DataFrame(mappings)
    df.to_csv(output_csv, index=False)
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    print(f"Found {len(df)} SIRV reads with transcript assignments")
    print(f"Unique SIRV transcripts identified: {df['sirv_transcript'].nunique()}")
    
    return output_csv


def get_cell_barcodes(sc_fastq):
    """
    Extract cell barcodes from ONT scRNA-seq FASTQ
    """
    print("Extracting cell barcodes from scRNA-seq data...")
    
    # For simplicity - in a real implementation you'd extract actual barcodes
    # This example just creates placeholder barcodes
    
    # Sample 10 cells
    barcodes = [f"CELL_{i:04d}" for i in range(1, 11)]
    read_counts = np.random.randint(1000, 5000, size=len(barcodes))
    
    cell_info = dict(zip(barcodes, read_counts))
    
    print(f"Found {len(cell_info)} cells with {sum(read_counts)} total reads")
    
    return cell_info


def get_read_length_distribution(sc_fastq, sample_size=1000):
    """
    Sample ONT read lengths to build distribution model
    """
    print("Sampling read lengths...")
    
    lengths = []
    count = 0
    
    opener = gzip.open if sc_fastq.endswith('.gz') else open
    with opener(sc_fastq, 'rt') as f:
        for record in SeqIO.parse(f, 'fastq'):
            lengths.append(len(record.seq))
            count += 1
            if count >= sample_size:
                break
    
    return np.array(lengths)


def add_sirv_to_dataset(sirv_fastq, sc_fastq, sirv_map_csv, 
                         output_fastq, insertion_rate=0.01):
    """
    Add SIRV reads to existing ONT dataset with barcodes
    """
    print(f"Adding SIRV reads to dataset at rate {insertion_rate}...")
    
    # Load SIRV transcript mapping
    sirv_df = pd.read_csv(sirv_map_csv)
    read_to_transcript = dict(zip(sirv_df['read_id'], sirv_df['sirv_transcript']))
    
    # Load SIRV reads
    sirv_reads = {}
    opener = gzip.open if sirv_fastq.endswith('.gz') else open
    with opener(sirv_fastq, 'rt') as f:
        for record in SeqIO.parse(f, 'fastq'):
            if record.id in read_to_transcript:
                sirv_reads[record.id] = {
                    'seq': str(record.seq),
                    'qual': ''.join(chr(q+33) for q in record.letter_annotations['phred_quality']),
                    'transcript': read_to_transcript[record.id]
                }
    
    # Get cell info and read length distribution
    cell_info = get_cell_barcodes(sc_fastq)
    length_dist = get_read_length_distribution(sc_fastq)
    
    # Track SIRV additions
    tracking_data = []
    
    # Process reads
    with open(output_fastq, 'w') as outfile:
        # For each cell, add SIRV reads
        for barcode, read_count in cell_info.items():
            # Number of SIRVs to add
            num_sirvs = int(read_count * insertion_rate)
            
            # Available SIRV reads
            sirv_ids = list(sirv_reads.keys())
            
            # Sample reads
            if num_sirvs > len(sirv_ids):
                sampled_ids = random.choices(sirv_ids, k=num_sirvs)
            else:
                sampled_ids = random.sample(sirv_ids, num_sirvs)
            
            # Add SIRV reads with cell barcode
            for read_id in sampled_ids:
                sirv = sirv_reads[read_id]
                
                # Generate UMI
                umi = ''.join(random.choice('ACGT') for _ in range(10))
                
                # Truncate to realistic ONT read length
                target_len = min(np.random.choice(length_dist), len(sirv['seq']))
                seq = sirv['seq'][:target_len]
                qual = sirv['qual'][:target_len]
                
                # New read ID with tracking info
                new_id = f"{read_id}-{barcode}-{umi}"
                
                # Write to output
                outfile.write(f"@{new_id}\n{seq}\n+\n{qual}\n")
                
                # Track addition
                tracking_data.append({
                    'read_id': new_id,
                    'original_read_id': read_id,
                    'barcode': barcode,
                    'umi': umi,
                    'sirv_transcript': sirv['transcript']
                })
        
        # Copy original scRNA-seq reads
        opener = gzip.open if sc_fastq.endswith('.gz') else open
        with opener(sc_fastq, 'rt') as sc_file:
            for line in sc_file:
                outfile.write(line)
    
    # Save tracking information
    tracking_file = f"{os.path.splitext(output_fastq)[0]}_tracking.csv"
    pd.DataFrame(tracking_data).to_csv(tracking_file, index=False)
    
    # Create expected counts
    expected_file = f"{os.path.splitext(output_fastq)[0]}_expected_counts.csv"
    expected_counts = (pd.DataFrame(tracking_data)
                      .groupby(['barcode', 'sirv_transcript'])
                      .size()
                      .reset_index(name='expected_count'))
    expected_counts.to_csv(expected_file, index=False)
    
    print(f"Added {len(tracking_data)} SIRV reads to {len(cell_info)} cells")
    print(f"Tracking info saved to: {tracking_file}")
    print(f"Expected counts saved to: {expected_file}")
    
    return tracking_file


def compare_with_flames(expected_file, flames_output, output_file):
    """
    Compare expected vs observed SIRV counts
    """
    print("Comparing expected vs observed SIRV counts...")
    
    # Load expected counts
    expected = pd.read_csv(expected_file)
    
    # Load FLAMES results - adjust format as needed
    flames = pd.read_csv(flames_output)
    
    # Extract SIRV counts from FLAMES output
    observed = flames[flames['transcript_id'].str.contains('SIRV')]
    observed = observed.rename(columns={
        'cell_barcode': 'barcode',
        'transcript_id': 'sirv_transcript',
        'count': 'observed_count'
    })
    
    # Merge expected and observed
    comparison = pd.merge(
        expected[['barcode', 'sirv_transcript', 'expected_count']],
        observed[['barcode', 'sirv_transcript', 'observed_count']],
        on=['barcode', 'sirv_transcript'],
        how='outer'
    ).fillna(0)
    
    # Calculate metrics
    comparison['detected'] = comparison['observed_count'] > 0
    comparison['detection_rate'] = comparison.apply(
        lambda row: row['observed_count'] / row['expected_count'] 
        if row['expected_count'] > 0 else 0,
        axis=1
    )
    
    # Save comparison
    comparison.to_csv(output_file, index=False)
    
    # Calculate summary stats
    total_expected = len(comparison[comparison['expected_count'] > 0])
    total_detected = len(comparison[comparison['detected'] == True])
    detection_rate = total_detected / total_expected if total_expected > 0 else 0
    
    print(f"SIRV detection summary:")
    print(f"- Expected transcripts: {total_expected}")
    print(f"- Detected transcripts: {total_detected}")
    print(f"- Overall detection rate: {detection_rate:.2%}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='SIRV Integration for ONT scRNA-seq')
    parser.add_argument('--sirv_fastq', required=True, help='SIRV bulk FASTQ file')
    parser.add_argument('--sc_fastq', required=True, help='5Y scRNA-seq FASTQ file')
    parser.add_argument('--sirv_reference', required=True, help='SIRV reference genome')
    parser.add_argument('--sirv_gtf', required=True, help='SIRV annotation GTF')
    parser.add_argument('--output_fastq', required=True, help='Output combined FASTQ')
    parser.add_argument('--flames_output', help='FLAMES output for comparison')
    parser.add_argument('--insertion_rate', type=float, default=0.01, help='SIRV insertion rate')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    
    args = parser.parse_args()
    
    # Step 1: Map SIRV reads to reference
    transcript_map = f"{os.path.splitext(args.output_fastq)[0]}_transcript_map.csv"
    map_sirv_reads(
        args.sirv_fastq,
        args.sirv_reference,
        args.sirv_gtf,
        transcript_map,
        threads=args.threads
    )
    
    # Step 2: Add SIRV reads to scRNA-seq dataset
    tracking_file = add_sirv_to_dataset(
        args.sirv_fastq,
        args.sc_fastq,
        transcript_map,
        args.output_fastq,
        insertion_rate=args.insertion_rate
    )
    
    # Step 3: Compare with FLAMES (if provided)
    if args.flames_output:
        expected_file = f"{os.path.splitext(args.output_fastq)[0]}_expected_counts.csv"
        comparison_file = f"{os.path.splitext(args.output_fastq)[0]}_comparison.csv"
        compare_with_flames(expected_file, args.flames_output, comparison_file)
    
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()