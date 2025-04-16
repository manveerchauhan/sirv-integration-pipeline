#!/usr/bin/env python3
"""
Evaluate Coverage Models

This script evaluates machine learning coverage models (Random Forest or XGBoost) 
on a given BAM file without performing SIRV integration.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from sirv_pipeline.ml_coverage_model import RandomForestCoverageModel, XGBoostCoverageModel
from sirv_pipeline.main import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ML Coverage Models")
    parser.add_argument("--bam-file", required=True, help="BAM file with aligned transcripts")
    parser.add_argument("--gtf-file", required=True, help="GTF file with transcript annotations")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--min-reads", type=int, default=5, help="Minimum reads per transcript (default: 5)")
    parser.add_argument("--length-bins", type=int, default=5, help="Number of length bins (default: 5)")
    parser.add_argument("--model-type", type=str, default="rf", choices=["rf", "xgboost"], 
                        help="Model type: random forest (rf) or XGBoost (xgboost) (default: rf)")
    
    # Shared model parameters
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees/estimators (default: 100)")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth of trees (default: None)")
    parser.add_argument("--min-samples-split", type=int, default=2, help="Minimum samples required to split (default: 2)")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="Minimum samples in a leaf (default: 1)")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    # XGBoost specific parameters
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for XGBoost (default: 0.1)")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio for XGBoost (default: 0.8)")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, 
                        help="Column subsample ratio for XGBoost (default: 0.8)")
    
    # Feature extraction parameters
    parser.add_argument("--feature-extraction", action="store_true", help="Extract sequence features (requires reference FASTA)")
    parser.add_argument("--reference-file", help="Reference FASTA file (required if feature-extraction is enabled)")
    
    # Other parameters
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--report-file", type=str, help="Path to save the evaluation report (Markdown format). Defaults to coverage_report.md in the output directory.")
    parser.add_argument("--skip-report", action="store_true", help="Skip generating the evaluation report")
    
    return parser.parse_args()

def create_evaluation_report(metrics_dir, plots_dir, output_file, model_type="rf", logger=None):
    """
    Create a comprehensive report of the coverage model evaluation.
    
    Args:
        metrics_dir: Directory containing metrics CSV files
        plots_dir: Directory containing plot files
        output_file: Path to save the Markdown report
        model_type: Type of model used ("rf" or "xgboost")
        logger: Optional logger object
    
    Returns:
        bool: True if report was successfully created, False otherwise
    """
    if logger:
        logger.info(f"Creating evaluation report at {output_file}")
    
    # Load metrics data
    metrics_file = os.path.join(metrics_dir, "model_metrics.csv")
    feature_file = os.path.join(metrics_dir, "feature_importances.csv")
    
    if not os.path.exists(metrics_file):
        if logger:
            logger.error(f"Metrics file not found at {metrics_file}")
        return False
    
    metrics = pd.read_csv(metrics_file)
    
    # Get model name for display
    model_name = "Random Forest" if model_type == "rf" else "XGBoost"
    
    # Start building the report
    with open(output_file, 'w') as f:
        f.write(f"# {model_name} Coverage Model Evaluation Report\n\n")
        
        # Add model performance metrics
        f.write("## Model Performance Metrics\n\n")
        f.write("### Training Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        if 'train_r2' in metrics.columns:
            f.write(f"| R² Score | {metrics['train_r2'].iloc[0]:.4f} |\n")
        if 'train_mse' in metrics.columns:
            f.write(f"| Mean Squared Error | {metrics['train_mse'].iloc[0]:.4f} |\n")
        if 'train_mae' in metrics.columns:
            f.write(f"| Mean Absolute Error | {metrics['train_mae'].iloc[0]:.4f} |\n")
        
        f.write("\n### Test Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        if 'test_r2' in metrics.columns:
            f.write(f"| R² Score | {metrics['test_r2'].iloc[0]:.4f} |\n")
        if 'test_mse' in metrics.columns:
            f.write(f"| Mean Squared Error | {metrics['test_mse'].iloc[0]:.4f} |\n")
        if 'test_mae' in metrics.columns:
            f.write(f"| Mean Absolute Error | {metrics['test_mae'].iloc[0]:.4f} |\n")
        
        # Add model-specific metrics
        if model_type == "rf" and 'oob_score' in metrics.columns:
            f.write(f"\n### Out-of-Bag Score: {metrics['oob_score'].iloc[0]:.4f}\n")
            f.write("\n*Out-of-bag score is an estimate of model accuracy computed on the observations not used for training.*\n")
        elif model_type == "xgboost" and 'best_iteration' in metrics.columns:
            f.write(f"\n### Best Iteration: {int(metrics['best_iteration'].iloc[0])}\n")
            f.write("\n*Best iteration is the number of boosting rounds at which the model performed best on validation data.*\n")
        
        # Add feature importances if available
        if os.path.exists(feature_file):
            feature_importances = pd.read_csv(feature_file)
            if not feature_importances.empty:
                f.write("\n## Top Feature Importances\n\n")
                f.write("| Feature | Importance |\n")
                f.write("|---------|------------|\n")
                
                # Get top 10 features
                top_features = feature_importances.nlargest(10, 'Importance')
                for _, row in top_features.iterrows():
                    f.write(f"| {row['Feature']} | {row['Importance']:.4f} |\n")
        
        # Add plots
        f.write("\n## Evaluation Plots\n\n")
        
        # Model evaluation plot
        eval_plot = os.path.join(plots_dir, "model_evaluation.png")
        if os.path.exists(eval_plot):
            f.write("### Model Evaluation\n\n")
            f.write(f"![Model Evaluation](plots/model_evaluation.png)\n\n")
        
        # Coverage bias plot
        bias_plot = os.path.join(plots_dir, "coverage_bias.png")
        if os.path.exists(bias_plot):
            f.write("### Coverage Bias Profile\n\n")
            f.write(f"![Coverage Bias](plots/coverage_bias.png)\n\n")
        
        # Feature importance plot
        importance_plot = os.path.join(plots_dir, "feature_importance.png")
        if os.path.exists(importance_plot):
            f.write("### Feature Importance\n\n")
            f.write(f"![Feature Importance](plots/feature_importance.png)\n\n")
        
        # Add summary and interpretation
        f.write("\n## Summary and Interpretation\n\n")
        
        if 'test_r2' in metrics.columns and 'train_r2' in metrics.columns:
            test_r2 = metrics['test_r2'].iloc[0]
            train_r2 = metrics['train_r2'].iloc[0]
            
            f.write(f"The {model_name.lower()} model achieved an R² score of {test_r2:.4f} on the test set ")
            f.write(f"and {train_r2:.4f} on the training set.\n\n")
            
            # Interpret R2 score
            if test_r2 > 0.9:
                f.write("The model shows excellent predictive performance with R² > 0.9, indicating it captures the coverage bias patterns very well.\n\n")
            elif test_r2 > 0.7:
                f.write("The model shows good predictive performance with R² > 0.7, indicating it captures most of the coverage bias patterns.\n\n")
            elif test_r2 > 0.5:
                f.write("The model shows moderate predictive performance with R² > 0.5, indicating it captures some of the coverage bias patterns.\n\n")
            else:
                f.write("The model shows limited predictive performance with R² < 0.5, indicating difficulty in capturing the coverage bias patterns.\n\n")
            
            # Check for overfitting
            if train_r2 - test_r2 > 0.2:
                f.write("**Note:** There appears to be significant overfitting, as the training R² is much higher than the test R².\n")
                f.write("Consider simplifying the model (e.g., reducing max_depth) or gathering more training data.\n\n")
        
        f.write("### Coverage Bias Interpretation\n\n")
        f.write("The coverage bias profile shows the relative coverage probability across transcript positions, ")
        f.write("where values greater than 1.0 indicate positions that are overrepresented in the sequencing data, ")
        f.write("and values less than 1.0 indicate underrepresented positions.\n\n")
        
        # Add date information
        f.write(f"\n\n*Report generated on {datetime.now().strftime('%Y-%m-%d')}*\n")
    
    if logger:
        logger.info(f"Successfully created evaluation report at {output_file}")
    
    return True

def create_readme(output_dir, bam_file, gtf_file, args, logger=None):
    """
    Create a README file with details about the evaluation.
    
    Args:
        output_dir: Base output directory
        bam_file: Path to BAM file used
        gtf_file: Path to GTF file used
        args: Command line arguments
        logger: Optional logger object
    """
    readme_file = os.path.join(output_dir, "README.md")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_name = "Random Forest" if args.model_type == "rf" else "XGBoost"
    results_dir = "rf_model_results" if args.model_type == "rf" else "xgb_model_results"
    
    with open(readme_file, 'w') as f:
        f.write(f"# {model_name} Coverage Model Evaluation\n\n")
        f.write(f"This directory contains a comprehensive evaluation of the {model_name.lower()} coverage bias model.\n")
        f.write(f"The evaluation was run on {timestamp}.\n\n")
        
        f.write("## Directory Structure\n\n")
        f.write(f"- `{results_dir}/`: Results from evaluating the {model_name.lower()} model\n")
        f.write("  - `plots/`: Visualizations of model performance and learned patterns\n")
        f.write("  - `metrics/`: CSV files with detailed metrics\n")
        f.write(f"  - `{args.model_type}_coverage_model.pkl`: The trained model\n\n")
        
        f.write("## Key Files\n\n")
        
        # Get the report file path relative to the base directory
        if not args.skip_report:
            report_path = args.report_file if args.report_file else os.path.join(output_dir, f"{args.model_type}_coverage_report.md")
            report_rel_path = os.path.relpath(report_path, output_dir)
            f.write(f"- `{report_rel_path}`: Comprehensive report of model performance\n")
        
        f.write("\n## Input Datasets\n\n")
        f.write(f"- FLAMES BAM: {bam_file}\n")
        f.write(f"- FLAMES GTF: {gtf_file}\n\n")
        
        f.write("## Model Parameters\n\n")
        f.write(f"- Model Type: {model_name}\n")
        f.write(f"- Min Reads: {args.min_reads}\n")
        f.write(f"- Length Bins: {args.length_bins}\n")
        f.write(f"- Trees (n_estimators): {args.n_estimators}\n")
        f.write(f"- Max Depth: {args.max_depth}\n")
        
        if args.model_type == "xgboost":
            f.write(f"- Learning Rate: {args.learning_rate}\n")
            f.write(f"- Subsample: {args.subsample}\n")
            f.write(f"- Column Sample: {args.colsample_bytree}\n")
        
        if args.feature_extraction:
            f.write(f"- Feature Extraction: Enabled\n")
            f.write(f"- Reference File: {args.reference_file}\n")
    
    if logger:
        logger.info(f"Created README file at {readme_file}")

def main():
    args = parse_args()
    
    # Validate arguments
    if args.feature_extraction and not args.reference_file:
        print("ERROR: --reference-file is required when --feature-extraction is enabled")
        return 1
    
    # Set default max depth if not specified
    if args.max_depth is None:
        if args.model_type == "rf":
            args.max_depth = 10  # Default for Random Forest
        else:
            args.max_depth = 6   # Default for XGBoost
    
    # Get model name and output directory name
    model_name = "Random Forest" if args.model_type == "rf" else "XGBoost"
    results_dir = "rf_model_results" if args.model_type == "rf" else "xgb_model_results"
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir = output_dir / results_dir
    Path(model_output_dir).mkdir(exist_ok=True)
    Path(model_output_dir / "plots").mkdir(exist_ok=True)
    Path(model_output_dir / "metrics").mkdir(exist_ok=True)
    
    # Set up logging
    log_file = output_dir / f"{args.model_type}_coverage_evaluation.log"
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(str(log_file), console_level=log_level, file_level=logging.DEBUG)
    
    # Log arguments
    logger.info(f"{model_name} Coverage Model Evaluation")
    logger.info(f"BAM file: {args.bam_file}")
    logger.info(f"GTF file: {args.gtf_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Minimum reads: {args.min_reads}")
    logger.info(f"Length bins: {args.length_bins}")
    logger.info(f"Trees (n_estimators): {args.n_estimators}")
    logger.info(f"Max depth: {args.max_depth}")
    logger.info(f"Threads: {args.threads}")
    
    if args.model_type == "xgboost":
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Subsample: {args.subsample}")
        logger.info(f"Column sample: {args.colsample_bytree}")
    
    if args.feature_extraction:
        logger.info(f"Feature extraction: Enabled")
        logger.info(f"Reference file: {args.reference_file}")
    
    # Create model configuration
    model_config = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
    }
    
    # Add XGBoost-specific parameters if needed
    if args.model_type == "xgboost":
        model_config.update({
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        })
    
    # Initialize the appropriate model
    logger.info(f"Initializing {model_name} coverage model")
    try:
        if args.model_type == "rf":
            model = RandomForestCoverageModel(
                seed=args.seed,
                logger=logger,
                config=model_config
            )
        else:
            model = XGBoostCoverageModel(
                seed=args.seed,
                logger=logger,
                config=model_config
            )
    except ImportError as e:
        logger.error(f"Failed to initialize model: {e}")
        if args.model_type == "xgboost":
            logger.error("XGBoost is required but not installed. Install with 'pip install xgboost'")
        return 1
    
    # Train the model
    logger.info(f"Training {model_name} model on BAM file")
    success = model.learn_from_bam(
        bam_file=args.bam_file,
        reference_file=args.reference_file if args.feature_extraction else None,
        annotation_file=args.gtf_file,
        min_reads=args.min_reads,
        feature_extraction=args.feature_extraction
    )
    
    if not success:
        logger.error("Failed to train model")
        return 1
    
    # Save model
    model_file = model_output_dir / f"{args.model_type}_coverage_model.pkl"
    logger.info(f"Saving trained model to {model_file}")
    model.save(str(model_file))
    
    # Save metrics
    metrics = model.get_performance_metrics()
    metrics_df = pd.DataFrame([metrics])
    metrics_file = model_output_dir / "metrics" / "model_metrics.csv"
    metrics_df.to_csv(str(metrics_file), index=False)
    
    # Save feature importances
    if model.feature_importances is not None and model.feature_names:
        importance_data = {
            "Feature": model.feature_names,
            "Importance": model.feature_importances
        }
        importance_df = pd.DataFrame(importance_data)
        importance_file = model_output_dir / "metrics" / "feature_importances.csv"
        importance_df.to_csv(str(importance_file), index=False)
    
    # Generate plots
    logger.info("Generating evaluation plots")
    
    # Performance evaluation plot
    eval_plot_file = model_output_dir / "plots" / "model_evaluation.png"
    model.plot_performance_evaluation(str(eval_plot_file))
    
    # Coverage bias plot
    bias_plot_file = model_output_dir / "plots" / "coverage_bias.png"
    model.plot_distributions(str(bias_plot_file))
    
    # Feature importance plot
    if model.feature_importances is not None and model.feature_names:
        importance_plot_file = model_output_dir / "plots" / "feature_importance.png"
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top features
        feature_importance = list(zip(model.feature_names, model.feature_importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_n = min(10, len(feature_importance))
        
        # Get top features for plotting
        features, importance = zip(*feature_importance[:top_n])
        y_pos = np.arange(len(features))
        
        # Create horizontal bar chart
        ax.barh(y_pos, importance, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        
        plt.tight_layout()
        plt.savefig(str(importance_plot_file), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create evaluation report
    if not args.skip_report:
        report_file = args.report_file if args.report_file else str(output_dir / f"{args.model_type}_coverage_report.md")
        logger.info(f"Creating evaluation report at {report_file}")
        create_evaluation_report(
            metrics_dir=str(model_output_dir / "metrics"),
            plots_dir=str(model_output_dir / "plots"),
            output_file=report_file,
            model_type=args.model_type,
            logger=logger
        )
    
    # Create README file
    create_readme(
        output_dir=str(output_dir),
        bam_file=args.bam_file,
        gtf_file=args.gtf_file,
        args=args,
        logger=logger
    )
    
    logger.info(f"{model_name} coverage model evaluation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 