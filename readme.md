# SIRV Integration Pipeline for Long-Read scRNA-seq

A pipeline for integrating SIRV spike-in reads into existing scRNA-seq datasets to benchmark isoform discovery tools.

![Pipeline Overview](pipeline_overview.png)

## Overview

This pipeline allows you to create semi-synthetic benchmarking datasets by adding SIRV (Spike-In RNA Variant) reads to an existing single-cell RNA-seq dataset. The primary purpose is to evaluate transcript discovery and quantification tools like FLAMES by providing a ground truth of known transcripts at controlled abundances.

Key features:
- Maps SIRV reads to reference to identify their transcript of origin
- Integrates SIRV reads into an existing ONT long-read scRNA-seq dataset
- Adds cell barcodes and UMIs to SIRV reads for single-cell analysis
- Truncates reads to match realistic length distributions
- Provides tracking information for expected vs. observed comparisons

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sirv-integration-pipeline.git
cd sirv-integration-pipeline

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- Python 3.6+
- minimap2
- samtools
- pysam
- pandas
- numpy
- Biopython

You can install most Python dependencies via pip:

```bash
pip install pysam pandas numpy biopython
```

For minimap2 and samtools, follow their installation instructions or use conda:

```bash
conda install -c bioconda minimap2 samtools
```

## Usage

### Basic Usage

```bash
python sirv_pipeline.py \
    --sirv_fastq /path/to/sirv_reads.fastq \
    --sc_fastq /path/to/5y_data.fastq \
    --sirv_reference /path/to/sirv_genome.fa \
    --sirv_gtf /path/to/sirv_annotation.gtf \
    --output_fastq /path/to/output/combined_data.fastq \
    --insertion_rate 0.01 \
    --threads 8
```

### After Running FLAMES Analysis

```bash
python sirv_pipeline.py \
    --sirv_fastq /path/to/sirv_reads.fastq \
    --sc_fastq /path/to/5y_data.fastq \
    --sirv_reference /path/to/sirv_genome.fa \
    --sirv_gtf /path/to/sirv_annotation.gtf \
    --output_fastq /path/to/output/combined_data.fastq \
    --flames_output /path/to/flames_results.csv \
    --insertion_rate 0.01 \
    --threads 8
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sirv_fastq` | Path to SIRV FASTQ file | *Required* |
| `--sc_fastq` | Path to scRNA-seq FASTQ file | *Required* |
| `--sirv_reference` | Path to SIRV reference genome | *Required* |
| `--sirv_gtf` | Path to SIRV annotation GTF | *Required* |
| `--output_fastq` | Path to output FASTQ file | *Required* |
| `--flames_output` | Path to FLAMES output for comparison | *Optional* |
| `--insertion_rate` | Proportion of SIRV reads to add | 0.01 (1%) |
| `--threads` | Number of threads for parallel processing | 4 |

## Pipeline Steps

### 1. Map SIRV reads to reference

The pipeline first maps the SIRV reads to the SIRV reference genome using minimap2 with settings optimized for ONT data. This step identifies which transcript each SIRV read represents.

```
minimap2 -ax map-ont -k 14 --secondary=no -t <threads> <sirv_reference> <sirv_fastq>
```

### 2. Identify SIRV transcripts

After mapping, the pipeline processes the alignment data to determine which SIRV transcript each read corresponds to. It uses the GTF annotation to find the best matching transcript for each read based on overlap.

The output is saved as `transcript_map.csv` with columns:
- `read_id`: Original SIRV read ID
- `sirv_transcript`: Corresponding SIRV transcript ID

### 3. Add SIRVs to scRNA-seq dataset

The pipeline integrates SIRV reads into the existing scRNA-seq dataset:
1. Examines the scRNA-seq data to determine cell barcodes and read counts
2. Samples SIRV read length distribution to create realistic truncation profiles
3. For each cell, adds ~1% SIRV reads (customizable with `--insertion_rate`)
4. Adds cell barcodes and UMIs to SIRV reads
5. Truncates reads to match realistic ONT read lengths
6. Combines modified SIRV reads with original dataset

This step produces:
- `combined_output.fastq`: The merged dataset containing original cells plus SIRV reads
- `tracking.csv`: Tracks which SIRV reads were added to which cells
- `expected_counts.csv`: Expected SIRV transcript counts per cell

### 4. Run FLAMES (External Step)

After generating the combined dataset, you need to run FLAMES (or your transcript discovery tool of choice) on the output FASTQ file. This step is not part of the pipeline and should be run separately according to FLAMES documentation.

### 5. Compare expected vs. observed (Optional)

If you provide FLAMES output with the `--flames_output` option, the pipeline will compare the expected SIRV counts with the observed counts from FLAMES:
1. Loads the expected counts generated in step 3
2. Loads the observed counts from FLAMES
3. Compares the two to calculate detection rates and accuracy metrics

The comparison results are saved to `comparison.csv` with columns:
- `barcode`: Cell barcode
- `sirv_transcript`: SIRV transcript ID
- `expected_count`: Number of reads added for this transcript
- `observed_count`: Number of reads detected by FLAMES
- `detected`: Whether the transcript was detected (1/0)
- `detection_rate`: Ratio of observed to expected count

## Output Files

| File | Description | Columns/Format |
|------|-------------|----------------|
| `transcript_map.csv` | SIRV read to transcript mapping | read_id, sirv_transcript |
| `combined_output.fastq` | Merged dataset with SIRVs | FASTQ format, read headers contain barcode+UMI |
| `tracking.csv` | Tracking of SIRV reads added | read_id, original_read_id, barcode, umi, sirv_transcript |
| `expected_counts.csv` | Expected transcript counts | barcode, sirv_transcript, expected_count |
| `comparison.csv` | Comparison with FLAMES results | barcode, sirv_transcript, expected_count, observed_count, detected, detection_rate |

## Customization

### Adding UMIs with Different Formats

If your scRNA-seq data uses a different UMI/barcode format, modify the `add_sirv_to_dataset()` function:

```python
# Generate a custom format barcode+UMI
new_id = f"{read_id}-{custom_barcode_format}-{custom_umi_format}"
```

### Adjusting Read Truncation

To customize how read truncation works:

```python
# Set a fixed length instead of sampling from distribution
target_len = 1000  # Fixed length in bp

# Or use a different distribution model
target_len = np.random.normal(1000, 200)  # Normal distribution
```

## Troubleshooting

### Common Issues

1. **minimap2 not found**: Ensure minimap2 is installed and in your PATH
   ```bash
   which minimap2
   ```

2. **Memory errors during alignment**: Reduce the number of threads or use a machine with more RAM
   ```bash
   --threads 2
   ```

3. **SIRV reads not being detected**: Check that your SIRV reference and GTF files match the SIRVs used
   ```bash
   # Validate GTF format
   grep "transcript_id" sirv_annotation.gtf | head
   ```

## Citation

If you use this pipeline in your research, please cite:

```
Author, A. (2025). SIRV Integration Pipeline for Long-Read scRNA-seq. GitHub repository.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue on the GitHub repository or contact [your-email@example.com].
