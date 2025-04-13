"""
Feature Cache Module for SIRV Integration Pipeline

This module provides functionality to cache and retrieve pre-computed features
for machine learning models, reducing computation time for repeated analysis.
"""

import os
import pickle
import logging
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple, Set
import numpy as np

logger = logging.getLogger(__name__)

class FeatureCache:
    """
    Cache for machine learning features to avoid recalculation.
    
    This class implements a persistent cache for storing and retrieving
    feature vectors and other computed values used in ML models.
    """
    
    def __init__(self, cache_file: Optional[str] = None, max_size: int = 10000):
        """
        Initialize the feature cache.
        
        Args:
            cache_file (Optional[str]): Path to the cache file for persistence
            max_size (int): Maximum number of entries in the cache
        """
        self.cache_file = cache_file
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "inserts": 0,
            "evictions": 0
        }
        
        # Load cache from file if provided
        if cache_file and os.path.exists(cache_file):
            self._load_from_file()
    
    def _generate_key(self, feature_type: str, params: Dict[str, Any]) -> str:
        """
        Generate a unique key for a feature calculation based on parameters.
        
        Args:
            feature_type (str): Type of feature
            params (Dict[str, Any]): Parameters used for calculation
            
        Returns:
            str: Unique key for cache lookup
        """
        # Convert params to a sorted, stable string representation
        param_str = str(sorted([(k, str(v)) for k, v in params.items()]))
        
        # Generate MD5 hash of the feature type and parameters
        key = hashlib.md5(f"{feature_type}:{param_str}".encode()).hexdigest()
        return key
    
    def _load_from_file(self) -> None:
        """
        Load cache from file.
        """
        try:
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                self.cache = data.get('cache', {})
                self.stats = data.get('stats', self.stats)
                
                # Initialize access times for existing cache entries
                current_time = time.time()
                self.access_times = {k: current_time for k in self.cache.keys()}
                
                logger.info(f"Loaded {len(self.cache)} cached features from {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load feature cache from file: {e}")
            self.cache = {}
            self.access_times = {}
    
    def save_to_file(self) -> None:
        """
        Save cache to file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.cache_file:
            logger.warning("No cache file specified, skipping save operation")
            return
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'cache': self.cache,
                    'stats': self.stats
                }, f)
                
            logger.info(f"Saved {len(self.cache)} cached features to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save feature cache to file: {e}")
    
    def _evict_if_needed(self) -> None:
        """
        Evict least recently used entries if cache exceeds maximum size.
        """
        if len(self.cache) > self.max_size:
            # Sort by access time (oldest first)
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
            
            # Calculate number of items to evict (remove 20% of max_size)
            evict_count = int(self.max_size * 0.2)
            
            # Evict oldest entries
            for i in range(evict_count):
                if i < len(sorted_keys):
                    key = sorted_keys[i][0]
                    if key in self.cache:
                        del self.cache[key]
                        del self.access_times[key]
                        self.stats["evictions"] += 1
            
            logger.info(f"Evicted {evict_count} entries from feature cache")
    
    def get(self, feature_type: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get a cached feature by type and parameters.
        
        Args:
            feature_type (str): Type of feature
            params (Dict[str, Any]): Parameters used for calculation
            
        Returns:
            Optional[Dict[str, Any]]: Cached feature or None if not found
        """
        key = self._generate_key(feature_type, params)
        
        if key in self.cache:
            # Update access time
            self.access_times[key] = time.time()
            self.stats["hits"] += 1
            return self.cache[key]
        
        self.stats["misses"] += 1
        return None
    
    def put(self, feature_type: str, params: Dict[str, Any], data: Dict[str, Any]) -> None:
        """
        Add a feature to the cache.
        
        Args:
            feature_type (str): Type of feature
            params (Dict[str, Any]): Parameters used for calculation
            data (Dict[str, Any]): Computed feature data
        """
        key = self._generate_key(feature_type, params)
        
        # Check if we need to evict entries
        self._evict_if_needed()
        
        # Store in cache
        self.cache[key] = data
        self.access_times[key] = time.time()
        self.stats["inserts"] += 1
    
    def clear(self) -> None:
        """
        Clear the cache.
        """
        self.cache = {}
        self.access_times = {}
        logger.info("Feature cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, int]: Cache statistics
        """
        stats = self.stats.copy()
        stats["size"] = len(self.cache)
        return stats


class FeatureHashmap:
    """
    Feature hashmap for efficient feature lookup.
    
    This class provides a more efficient way to look up features by
    transcript ID, position, or other attributes.
    """
    
    def __init__(self):
        """Initialize the feature hashmap."""
        self.transcript_features: Dict[str, Dict[str, Any]] = {}
        self.position_features: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.global_features: Dict[str, Any] = {}
    
    def add_transcript_features(self, transcript_id: str, features: Dict[str, Any]) -> None:
        """
        Add features for a transcript.
        
        Args:
            transcript_id (str): Transcript ID
            features (Dict[str, Any]): Feature dictionary
        """
        self.transcript_features[transcript_id] = features
    
    def add_position_features(self, transcript_id: str, position: int, features: Dict[str, Any]) -> None:
        """
        Add features for a specific position in a transcript.
        
        Args:
            transcript_id (str): Transcript ID
            position (int): Position within the transcript
            features (Dict[str, Any]): Feature dictionary
        """
        self.position_features[(transcript_id, position)] = features
    
    def add_global_feature(self, feature_name: str, value: Any) -> None:
        """
        Add a global feature that applies to the entire dataset.
        
        Args:
            feature_name (str): Feature name
            value (Any): Feature value
        """
        self.global_features[feature_name] = value
    
    def get_transcript_features(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """
        Get features for a transcript.
        
        Args:
            transcript_id (str): Transcript ID
            
        Returns:
            Optional[Dict[str, Any]]: Feature dictionary or None if not found
        """
        return self.transcript_features.get(transcript_id)
    
    def get_position_features(self, transcript_id: str, position: int) -> Optional[Dict[str, Any]]:
        """
        Get features for a specific position in a transcript.
        
        Args:
            transcript_id (str): Transcript ID
            position (int): Position within the transcript
            
        Returns:
            Optional[Dict[str, Any]]: Feature dictionary or None if not found
        """
        return self.position_features.get((transcript_id, position))
    
    def get_global_feature(self, feature_name: str) -> Optional[Any]:
        """
        Get a global feature.
        
        Args:
            feature_name (str): Feature name
            
        Returns:
            Optional[Any]: Feature value or None if not found
        """
        return self.global_features.get(feature_name)
    
    def get_all_transcript_ids(self) -> Set[str]:
        """
        Get all transcript IDs in the hashmap.
        
        Returns:
            Set[str]: Set of transcript IDs
        """
        return set(self.transcript_features.keys())
    
    def get_all_positions(self, transcript_id: str) -> List[int]:
        """
        Get all positions for a transcript in the hashmap.
        
        Args:
            transcript_id (str): Transcript ID
            
        Returns:
            List[int]: List of positions
        """
        positions = [pos for (tid, pos) in self.position_features.keys() if tid == transcript_id]
        return sorted(positions)
    
    def to_numpy(self, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert transcript features to a numpy array.
        
        Args:
            feature_names (List[str]): List of feature names to include
            
        Returns:
            Tuple[np.ndarray, List[str]]: Feature array and list of transcript IDs
        """
        transcript_ids = list(self.transcript_features.keys())
        n_samples = len(transcript_ids)
        n_features = len(feature_names)
        
        # Initialize feature array
        X = np.zeros((n_samples, n_features))
        
        # Fill feature array
        for i, tid in enumerate(transcript_ids):
            features = self.transcript_features[tid]
            for j, feature_name in enumerate(feature_names):
                if feature_name in features:
                    X[i, j] = features[feature_name]
        
        return X, transcript_ids
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save hashmap to file.
        
        Args:
            file_path (str): Path to save the hashmap file
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'transcript_features': self.transcript_features,
                    'position_features': self.position_features,
                    'global_features': self.global_features
                }, f)
                
            logger.info(f"Saved feature hashmap to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save feature hashmap to file: {e}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'FeatureHashmap':
        """
        Load hashmap from file.
        
        Args:
            file_path (str): Path to the hashmap file
            
        Returns:
            FeatureHashmap: Loaded feature hashmap
            
        Raises:
            ValueError: If the file cannot be read
        """
        if not os.path.exists(file_path):
            raise ValueError(f"Feature hashmap file not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            hashmap = cls()
            hashmap.transcript_features = data.get('transcript_features', {})
            hashmap.position_features = data.get('position_features', {})
            hashmap.global_features = data.get('global_features', {})
            
            logger.info(f"Loaded feature hashmap from {file_path} with {len(hashmap.transcript_features)} transcripts")
            return hashmap
        except Exception as e:
            logger.error(f"Failed to load feature hashmap from file: {e}")
            raise ValueError(f"Failed to load feature hashmap from {file_path}: {str(e)}") 