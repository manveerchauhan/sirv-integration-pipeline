# RF Coverage Model Bug Fixes

## Issues Identified

The random forest coverage model evaluation was showing several issues:

1. **Very small MSE/MAE values**: The MSE and MAE values were displaying as 0.0000 in the report (while actually being around 10^-5).
2. **Negative R² scores**: The test R² score of -0.3224 and OOB score of -0.2202 indicated the model was performing worse than a simple mean predictor.
3. **Zero importance for GC content**: The feature importance for gc_content was 0.0000, suggesting this biologically important feature wasn't contributing to predictions.
4. **Insufficient features**: The model was only using basic features without sequence-specific information.
5. **Overfitting**: The large gap between training and testing performance indicated overfitting to noise.

## Changes Made

The following changes were implemented to fix these issues:

### 1. Enabled Feature Extraction
Added `--feature-extraction` flag and `--reference-file` parameter to extract important sequence-based features like GC content, structure potential, and more detailed transcript properties.

```bash
--feature-extraction
--reference-file ${REFERENCE_FASTA}
```

### 2. Fixed Normalization
Modified the normalization in `_extract_coverage_data` to prevent extremely small values and improve the scale of predictions:

```python
# Before:
binned_coverage = binned_coverage / np.mean(binned_coverage)

# After:
binned_coverage = 100.0 * binned_coverage / (np.mean(binned_coverage) + 1e-10)
```

This change:
- Scales values by 100x to make them more meaningful
- Adds a small epsilon (1e-10) to prevent division by very small numbers

### 3. Increased Min Reads Threshold
Changed `--min-reads` from 5 to 15 to ensure more reliable coverage patterns:

```bash
--min-reads 15
```

### 4. Reduced Model Complexity
Lowered the `--max-depth` from 10 to 5 to reduce overfitting:

```bash
--max-depth 5
```

## Expected Improvements

These changes should result in:

1. More meaningful MSE and MAE values (not showing as 0.0000)
2. Positive R² scores on test data
3. Better feature importance distribution with GC content contributing meaningfully
4. Improved generalization with reduced gap between training and testing performance
5. More biologically relevant coverage patterns

## How to Use

Run the fixed script with:

```bash
sbatch rf_coverage_eval.slurm
```

The output will be in a new directory: `/data/gpfs/projects/punim2251/rf_coverage_eval_fixed_[TIMESTAMP]` 