"""
Model Configuration Module for SIRV Integration Pipeline

This module handles configuration parameters for machine learning models used in the
SIRV integration pipeline, specifically for the random forest coverage bias model.

It provides functionality to:
1. Load and save configuration from/to JSON or YAML files
2. Access predefined configuration presets for different sequencing technologies
3. Validate configuration parameters
"""

import os
import json
import logging
import yaml
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Default configuration for random forest model
DEFAULT_CONFIG = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "feature_importance_method": "permutation"
}

# Configuration presets for different sequencing technologies
CONFIG_PRESETS = {
    "nanopore_cdna": {
        "description": "Optimized for Oxford Nanopore cDNA sequencing",
        "model_type": "random_forest",
        "parameters": {
            "n_estimators": 150,
            "max_depth": 8,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "feature_importance_method": "permutation"
        },
        "feature_weights": {
            "gc_content": 2.0,
            "sequence_complexity": 1.5,
            "secondary_structure": 1.0
        }
    },
    "pacbio_isoseq": {
        "description": "Optimized for PacBio IsoSeq sequencing",
        "model_type": "random_forest",
        "parameters": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "feature_importance_method": "permutation"
        },
        "feature_weights": {
            "gc_content": 1.0,
            "sequence_complexity": 1.0,
            "secondary_structure": 1.5
        }
    },
    "direct_rna": {
        "description": "Optimized for direct RNA sequencing",
        "model_type": "random_forest",
        "parameters": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 4,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "bootstrap": True,
            "feature_importance_method": "permutation"
        },
        "feature_weights": {
            "gc_content": 1.5,
            "sequence_complexity": 2.0,
            "secondary_structure": 2.5
        }
    },
    "balanced": {
        "description": "Balanced configuration suitable for various technologies",
        "model_type": "random_forest",
        "parameters": {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_split": 3,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "feature_importance_method": "permutation"
        },
        "feature_weights": {
            "gc_content": 1.0,
            "sequence_complexity": 1.0,
            "secondary_structure": 1.0
        }
    }
}


def get_available_presets() -> List[Dict[str, Any]]:
    """
    Returns a list of available configuration presets with their descriptions.
    
    Returns:
        List[Dict[str, Any]]: List of preset information
    """
    presets = []
    for name, config in CONFIG_PRESETS.items():
        presets.append({
            "name": name,
            "description": config.get("description", ""),
            "model_type": config.get("model_type", "")
        })
    return presets


def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        file_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        ValueError: If the file format is not supported or the file cannot be read
    """
    if not os.path.exists(file_path):
        raise ValueError(f"Configuration file not found: {file_path}")
    
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        if ext == '.json':
            with open(file_path, 'r') as f:
                config = json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
        
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        raise ValueError(f"Failed to load configuration from {file_path}: {str(e)}")


def save_config_to_file(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration to a JSON or YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        file_path (str): Path to save the configuration file
        
    Raises:
        ValueError: If the file format is not supported or the file cannot be written
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        if ext == '.json':
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif ext in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
            
        logger.info(f"Configuration saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving configuration file: {e}")
        raise ValueError(f"Failed to save configuration to {file_path}: {str(e)}")


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """
    Get configuration preset by name.
    
    Args:
        preset_name (str): Name of the configuration preset
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        ValueError: If the preset name is not found
    """
    if preset_name not in CONFIG_PRESETS:
        available_presets = ", ".join(CONFIG_PRESETS.keys())
        raise ValueError(f"Configuration preset '{preset_name}' not found. Available presets: {available_presets}")
    
    return CONFIG_PRESETS[preset_name]


def get_model_config(preset_name: Optional[str] = None, config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Get model configuration based on preset name or configuration file.
    
    Priority order:
    1. Configuration file (if provided)
    2. Preset name (if provided)
    3. Default configuration
    
    Args:
        preset_name (Optional[str]): Name of the configuration preset
        config_file (Optional[str]): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Apply preset configuration if provided
    if preset_name:
        preset_config = get_preset_config(preset_name)
        
        # Update config with preset parameters
        if "parameters" in preset_config:
            config.update(preset_config["parameters"])
        
        # Add additional preset-specific fields
        for key, value in preset_config.items():
            if key not in ["model_type", "parameters", "description"]:
                config[key] = value
    
    # Load configuration from file if provided (highest priority)
    if config_file:
        file_config = load_config_from_file(config_file)
        config.update(file_config)
    
    return config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration parameters for random forest model.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Dict[str, Any]: Validated configuration dictionary
    """
    # Ensure essential parameters have valid values
    validated = config.copy()
    
    # Validate n_estimators
    if "n_estimators" in validated:
        validated["n_estimators"] = max(10, int(validated["n_estimators"]))
    else:
        validated["n_estimators"] = DEFAULT_CONFIG["n_estimators"]
    
    # Validate max_depth
    if "max_depth" in validated:
        validated["max_depth"] = max(1, int(validated["max_depth"]))
    else:
        validated["max_depth"] = DEFAULT_CONFIG["max_depth"]
    
    # Add other validations as needed
    
    return validated 