# SIRV Integration Pipeline for Long-Read scRNA-seq

A pipeline for integrating SIRV spike-in reads into existing scRNA-seq datasets to benchmark isoform discovery tools.

## Overview

This pipeline allows you to create semi-synthetic benchmarking datasets by adding SIRV (Spike-In RNA Variant) reads to an existing single-cell RNA-seq dataset. The primary purpose is to evaluate transcript discovery and quantification tools like FLAMES by providing a ground truth of known transcripts at controlled abundances.

Key features:
- Maps SIRV reads to reference to identify their transcript of origin
- Integrates SIRV reads into an existing ONT long-read scRNA-seq dataset
- Adds cell barcodes and UMIs to SIRV reads for single-cell analysis
- Models read length distributions to match the target dataset
- **NEW: Models 5'-3' coverage bias** to match the sequencing protocol
- Provides tracking information for expected vs. observed comparisons

## Installation

```bash
# Clone the repository
git clone https://github.com/manveerchauhan/sirv-integration-pipeline.git
cd sirv-integration-pipeline

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

Python 3.7+, numpy, pandas, pysam, biopython, matplotlib, jinja2
External: minimap2, samtools

You can install most Python dependencies via pip:

```bash
pip install pysam pandas numpy biopython matplotlib
```

For minimap2 and samtools, follow their installation instructions or use conda:

```bash
conda install -c bioconda minimap2 samtools
```

## Usage

### Basic Usage

```bash
sirv-pipeline \
    --sirv_fastq sirv_reads.fastq \
    --sc_fastq scRNA_data.fastq \
    --sirv_reference sirv_genome.fa \
    --sirv_gtf sirv_annotation.gtf \
    --reference_transcriptome reference.fa \
    --output_fastq combined_data.fastq \
    --insertion_rate 0.01
```

## The 5'-3' Coverage Bias Model

Different sequencing platforms and library preparation methods result in distinct patterns of where along the transcript reads tend to start and end. For example:

- Some protocols favor the 5' end of transcripts
- Others favor the 3' end 
- Some produce more uniform coverage
- Some have internal priming leading to truncated reads

The new coverage bias modeling learns these patterns from the original scRNA-seq dataset and applies similar biases to the SIRV reads being inserted. This is crucial for creating realistic benchmarking datasets, as these coverage patterns affect isoform detection and quantification.

### How Coverage Bias Modeling Works

1. **Learning Phase**: 
   - Aligns a sample of existing scRNA-seq reads to the reference transcriptome
   - Records where along transcripts the reads tend to start and end (as normalized positions)
   - Creates probability distributions for read start and end positions

2. **Application Phase**:
   - When adding a SIRV read, samples start and end positions based on the learned distributions
   - Extracts the corresponding fragment from the transcript
   - Ensures the fragment has an appropriate length (based on learned read length distribution)

### Enabling Coverage Bias Modeling

To use the 5'-3' coverage bias modeling, provide the reference transcriptome:

```bash
sirv-pipeline \
    --sirv_fastq sirv_reads.fastq \
    --sc_fastq scRNA_data.fastq \
    --sirv_reference sirv_genome.fa \
    --sirv_gtf sirv_annotation.gtf \
    --reference_transcriptome reference.fa \  # Required for coverage modeling
    --output_fastq combined_data.fastq
```

To disable coverage bias modeling:

```bash
sirv-pipeline \
    --sirv_fastq sirv_reads.fastq \
    --sc_fastq scRNA_data.fastq \
    --sirv_reference sirv_genome.fa \
    --sirv_gtf sirv_annotation.gtf \
    --output_fastq combined_data.fastq \
    --no_coverage_bias
```

### Visualizing the Coverage Bias Model

The pipeline automatically generates a visualization of the learned coverage bias model, showing the probability distributions for read start and end positions. This can be useful for understanding the characteristics of your scRNA-seq dataset.

The plot is saved to `output_dir/coverage_bias.png`.

## Examples

### Example with Coverage Bias Modeling

The repository includes an example script demonstrating 5'-3' coverage bias modeling:

```bash
python examples/coverage_bias_example.py \
    --sirv_fastq sirv_reads.fastq \
    --sc_fastq scRNA_data.fastq \
    --sirv_reference sirv_genome.fa \
    --sirv_gtf sirv_annotation.gtf \
    --reference_transcriptome reference.fa \
    --output_dir ./coverage_example
```

This script:
1. Creates a coverage bias model from the scRNA-seq dataset
2. Generates synthetic data both with and without the coverage bias model
3. Compares the results to show the impact of coverage bias modeling
4. Produces visualizations of the differences

## Advanced Settings

### Coverage Bias Model Parameters

You can tune the coverage bias model parameters using these command-line options:

```bash
--sample_size INT       Number of reads to sample for modeling (default: 1000)
--min_overlap FLOAT     Minimum overlap for transcript assignment (default: 0.5)
--no_coverage_bias      Disable 5'-3' coverage bias modeling
```

### Other Advanced Options

```bash
--insertion_rate FLOAT  SIRV insertion rate (0-0.5, default: 0.01)
--threads INT           Number of threads to use (default: 4)
--seed INT              Random seed for reproducibility
--keep_temp             Keep temporary files for debugging
--verbose               Enable verbose logging
```

## The Complete Pipeline

The complete SIRV integration pipeline workflow:

1. **Map SIRV Reads**: Map SIRV reads to reference to identify transcript of origin
2. **Learn Read Length Distribution**: Learn read length distribution from scRNA-seq data
3. **NEW: Learn Coverage Bias Model**: Learn 5'-3' coverage bias from scRNA-seq data
4. **Add SIRV Reads**: Add SIRV reads with cell barcodes, realistic lengths, and coverage patterns
5. **Evaluate with FLAMES**: Compare expected vs. observed transcripts from FLAMES output

## Citing SynthLongRead

If you use SIRV Integration Pipeline in your research, please cite:

```
[Citation information will be added after publication]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
Feel free to contact me for questions/suggestions: mschauhan@student.unimelb.edu.au

## Pipeline Schematic [WIP]
![Pipeline Schematic](pipeline_overview.png)
