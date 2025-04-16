# SIRV Integration Pipeline Bug Fix

## Issue: Polynomial Feature Dimension Mismatch

The SIRV integration pipeline was encountering the following error during coverage prediction:

```
Error predicting coverage for position 0.0: X has 17 features, but PolynomialFeatures is expecting 4 features as input.
```

## Root Cause

The ML coverage model in `ml_coverage_model.py` was using a scikit-learn Pipeline with a PolynomialFeatures transformer followed by a regressor (either RandomForest or GradientBoosting). 

The problem occurred because:
1. The model was trained with a certain number of input features during training
2. When making predictions later, a different number of features (17 instead of 4) was being provided
3. The PolynomialFeatures transformer has a fixed expectation of the input dimension and cannot automatically handle varying numbers of input features

## Solution

The following changes were made to fix the issue:

1. Modified `ml_coverage_model.py` to use the regressor directly without the PolynomialFeatures transformer
   - Removed the Pipeline structure and directly used the base model
   - Updated feature importance extraction to work directly with the model
   - Fixed prediction code to avoid assuming a pipeline structure

2. Updated `run_pipeline_fixed.py` to:
   - Add a new command-line flag `--disable-polynomial-features` 
   - Set an environment variable when this flag is used
   - Add appropriate warnings about ML model issues

3. Updated `sirv_model_comparison.slurm` to use the new flag for both model types

## Usage

When running the SIRV integration pipeline with ML-based coverage models, add the `--disable-polynomial-features` flag:

```bash
python run_pipeline_fixed.py --coverage-model random_forest --disable-polynomial-features
```

Or when using the SLURM script:

```bash
sbatch sirv_model_comparison.slurm
```

## Long-term Fix

For a more robust long-term solution, consider:

1. Standardizing feature extraction to ensure consistent dimensions
2. Adding a preprocessing step that handles variable feature dimensions
3. Creating a more flexible model architecture that can adapt to changing feature dimensions

## Date Fixed

This issue was fixed on [current date] 