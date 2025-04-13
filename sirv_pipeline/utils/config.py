"""
Configuration utilities for the SIRV Integration Pipeline.

This module provides functions for loading and managing configuration
presets for the coverage bias models, making it easier for users to
customize parameters for their specific data types.
"""

import os
import json
import logging
import yaml
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default model parameter presets for different RNA-seq protocols
DEFAULT_PRESETS = {
    "nanopore_cdna": {
        "model_type": "random_forest",
        "model_params": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 5,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": True,
            "ccp_alpha": 0.001
        },
        "feature_weights": {
            "gc_content": 1.5,
            "transcript_length": 1.2,
            "five_prime_gc": 1.4,
            "hairpin_potential": 1.1
        },
        "description": "Optimized for Oxford Nanopore cDNA data with typical 3' bias"
    },
    "pacbio_isoseq": {
        "model_type": "random_forest",
        "model_params": {
            "n_estimators": 150,
            "max_depth": 6,
            "min_samples_split": 8,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": True,
            "ccp_alpha": 0.002
        },
        "feature_weights": {
            "gc_content": 1.2,
            "five_prime_gc": 1.3,
            "three_prime_gc": 1.2,
            "hairpin_potential": 1.4
        },
        "description": "Optimized for PacBio IsoSeq data with more balanced coverage"
    },
    "direct_rna": {
        "model_type": "random_forest",
        "model_params": {
            "n_estimators": 250,
            "max_depth": 10,
            "min_samples_split": 4,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": True,
            "ccp_alpha": 0.0005
        },
        "feature_weights": {
            "gc_content": 1.3,
            "transcript_length": 1.5,
            "five_prime_gc": 1.8,
            "g_quadruplex_potential": 1.6
        },
        "description": "Optimized for direct RNA sequencing with 5' bias focus"
    },
    "balanced": {
        "model_type": "random_forest",
        "model_params": {
            "n_estimators": 180,
            "max_depth": 7,
            "min_samples_split": 6,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": True,
            "ccp_alpha": 0.001
        },
        "feature_weights": {
            "gc_content": 1.0,
            "transcript_length": 1.0,
            "five_prime_gc": 1.0,
            "three_prime_gc": 1.0,
            "hairpin_potential": 1.0,
            "g_quadruplex_potential": 1.0,
            "complexity": 1.0,
            "gc_skew": 1.0
        },
        "description": "Balanced parameters with equal feature weights"
    }
}

def load_model_config(preset: Optional[str] = None, config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load model parameters from a preset configuration or custom file.
    
    Args:
        preset: Preset config type ('nanopore_cdna', 'pacbio_isoseq', 'direct_rna', 'balanced')
        config_file: Path to custom JSON/YAML config file
    
    Returns:
        Dict with model parameters
    
    Examples:
        >>> config = load_model_config('nanopore_cdna')
        >>> config = load_model_config(config_file='my_config.json')
    """
    # First, check if a custom config file is provided
    if config_file and os.path.exists(config_file):
        try:
            # Load custom config from file
            with open(config_file) as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    try:
                        import yaml
                        config = yaml.safe_load(f)
                        logger.info(f"Loaded custom YAML config from {config_file}")
                        return config
                    except ImportError:
                        logger.warning("PyYAML not installed, falling back to JSON parser")
                        import json
                        config = json.load(f)
                        logger.info(f"Loaded custom JSON config from {config_file}")
                        return config
                else:
                    # Default to JSON for other extensions
                    import json
                    config = json.load(f)
                    logger.info(f"Loaded custom JSON config from {config_file}")
                    return config
        except Exception as e:
            logger.error(f"Failed to load custom config from {config_file}: {str(e)}")
            logger.warning(f"Falling back to default preset")
    
    # If preset is specified and valid, use it
    if preset in DEFAULT_PRESETS:
        logger.info(f"Using preset configuration: {preset}")
        return DEFAULT_PRESETS[preset]
    
    # Otherwise, use nanopore_cdna as default
    default_preset = "nanopore_cdna"
    logger.info(f"Using default preset configuration: {default_preset}")
    return DEFAULT_PRESETS[default_preset]

def save_model_config(config: Dict[str, Any], output_file: str) -> bool:
    """
    Save a model configuration to a file.
    
    Args:
        config: Configuration dictionary to save
        output_file: Path to save the configuration
    
    Returns:
        True if successful, False otherwise
    
    Examples:
        >>> config = load_model_config('nanopore_cdna')
        >>> config['model_params']['n_estimators'] = 300
        >>> save_model_config(config, 'my_custom_config.json')
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save based on file extension
        if output_file.endswith('.yaml') or output_file.endswith('.yml'):
            try:
                import yaml
                with open(output_file, 'w') as f:
                    yaml.dump(config, f, sort_keys=False, indent=4)
                logger.info(f"Saved configuration to YAML file: {output_file}")
                return True
            except ImportError:
                logger.warning("PyYAML not installed, falling back to JSON format")
                # Change extension to .json
                output_file = os.path.splitext(output_file)[0] + '.json'
                
        # Default to JSON
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved configuration to JSON file: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {output_file}: {str(e)}")
        return False

def get_available_presets() -> Dict[str, str]:
    """
    Get the list of available preset configurations with descriptions.
    
    Returns:
        Dictionary of preset names and their descriptions
    """
    return {name: preset.get('description', 'No description available') 
            for name, preset in DEFAULT_PRESETS.items()}

def print_preset_info(preset_name: str) -> None:
    """
    Print detailed information about a specific preset.
    
    Args:
        preset_name: Name of the preset to print info for
    """
    if preset_name not in DEFAULT_PRESETS:
        print(f"Preset '{preset_name}' not found. Available presets are:")
        for name, desc in get_available_presets().items():
            print(f"- {name}: {desc}")
        return
    
    preset = DEFAULT_PRESETS[preset_name]
    print(f"Preset: {preset_name}")
    print(f"Description: {preset.get('description', 'No description available')}")
    print("\nModel Parameters:")
    for key, value in preset.get('model_params', {}).items():
        print(f"  {key}: {value}")
    
    if 'feature_weights' in preset:
        print("\nFeature Weights:")
        for feature, weight in preset['feature_weights'].items():
            print(f"  {feature}: {weight}") 