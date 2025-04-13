# Random Forest Coverage Model for RNA-seq Data

This document explains how to use the Random Forest machine learning model for coverage bias modeling in the SIRV Integration Pipeline.

## Overview

The `ml_random_forest` model type provides an advanced approach to modeling transcript coverage bias using Random Forest regression. This model is specifically optimized for RNA-seq data with the following benefits:

- Better captures complex, non-linear coverage patterns across transcripts
- Takes into account RNA-specific features like secondary structure, GC content, and homopolymer regions
- Provides feature importance ranking to understand what drives coverage bias
- More robust to noise and outliers than parametric models
- Can generalize well to unseen transcripts

## When to Use the Random Forest Model

Consider using the `ml_random_forest` model when:

1. Your dataset shows complex coverage bias patterns not well-captured by default models
2. You have sufficient training data (BAM files with reads aligned to transcripts)
3. You need more accurate coverage modeling for specific RNA-seq protocols
4. Coverage bias varies significantly across different transcript types
5. You want to understand what sequence features are driving the coverage bias

## Features Used by the Model

The random forest model utilizes a rich set of RNA-seq specific features:

- **Position features**: Relative position along transcript (5' → 3')
- **Transcript length**: Log-transformed to handle the wide range of transcript lengths
- **GC content**: Overall and at 5'/3' ends of transcripts
- **RNA structure features**: 
  - Hairpin potential
  - G-quadruplex potential 
  - Position-dependent structural effects
- **Sequence complexity**: Affects mappability and coverage uniformity
- **Homopolymer and repeat regions**: Known to affect long-read sequencing accuracy
- **GC skew**: Important for strand-specific protocols

## Command-line Usage

To use the random forest model, specify the `--coverage-model ml_random_forest` argument along with a BAM file for training:

```bash
python -m sirv_pipeline \
  --integration \
  --output-dir /path/to/output \
  --sirv-fastq /path/to/sirv.fastq \
  --sc-fastq /path/to/sc_reads.fastq \
  --sirv-reference /path/to/sirv_reference.fa \
  --sirv-gtf /path/to/sirv_annotation.gtf \
  --coverage-model ml_random_forest \
  --learn-coverage-from /path/to/aligned_reads.bam \
  --annotation-file /path/to/transcript_annotation.gtf \
  --min-reads-for-learning 100 \
  --visualize-coverage
```

For best results, ensure that:

1. The BAM file contains enough reads (at least 100 per transcript)
2. The annotation file correctly defines transcript boundaries
3. A matching reference FASTA file is available for feature extraction

## Example Workflow

1. **Prepare training data**: Align RNA-seq reads to a reference transcriptome
   ```bash
   minimap2 -ax map-ont reference.fa reads.fastq > aligned_reads.bam
   samtools sort -o aligned_reads.sorted.bam aligned_reads.bam
   samtools index aligned_reads.sorted.bam
   ```

2. **Run the pipeline with random forest model**:
   ```bash
   python -m sirv_pipeline \
     --integration \
     --output-dir sirv_integration \
     --sirv-fastq sirv_reads.fastq \
     --sc-fastq sc_reads.fastq \
     --sirv-reference sirv_reference.fa \
     --coverage-model ml_random_forest \
     --learn-coverage-from aligned_reads.sorted.bam \
     --annotation-file transcripts.gtf \
     --visualize-coverage
   ```

3. **Examine the model visualization**:
   Open the generated `improved_coverage_bias.png` file to see the learned coverage patterns.

## Performance Considerations

The random forest model has been optimized for performance with the following settings:

- 200 estimators (trees) to balance accuracy and training speed
- Controlled depth to prevent overfitting to noise in coverage data
- Bootstrap sampling with out-of-bag scoring to evaluate performance
- Parallel processing using all available cores
- Memory-efficient feature representation

For very large datasets, increase the `--min-reads-for-learning` parameter to select a representative subset of transcripts for training.

## Comparing with Other Model Types

| Model Type | Use Case | Strengths | Limitations |
|------------|----------|-----------|-------------|
| `10x_cdna` | Default for 10x scRNA-seq | Fast, no training required | May not capture complex patterns |
| `direct_rna` | ONT direct RNA-seq | Fast, no training required | May not capture protocol-specific patterns |
| `custom` | Simple, data-driven model | Learns from data, computationally efficient | Uses simpler parametric model |
| `ml_gradient_boosting` | Complex coverage patterns | Learns complex patterns, high accuracy | Resource intensive, may overfit |
| `ml_random_forest` | RNA-specific patterns | Robust to noise, captures complex RNA features | Training requires good coverage data |

## Further Information

For more details on the implementation and parameters, see the source code in:
- `ml_coverage_model.py`: Implementation of the ML model
- `coverage_bias.py`: Integration with the coverage bias module 