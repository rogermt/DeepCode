"""
Experiment Tracking Abstraction Layer

This module provides a unified interface for experiment tracking across multiple backends,
enabling flexible observability without tying users to specific tracking services.

Key features:
- Abstract base class for consistent tracking interface
- Multiple backend implementations (Weave, W&B, Console)
- Graceful fallback when tracking services are unavailable
- Type-safe interface with comprehensive logging support
"""

import os
import sys
import yaml
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass

# Import tracking libraries
import weave
import wandb
from utils.wandb_utils import initialize_weave, initialize_wandb_run, upload_artifact


@dataclass
class AttrDict(dict):
    """Dictionary that allows attribute access to keys"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def load_experiment_config() -> Dict[str, Any]:
    """
    Load experiment tracking configuration from mcp_agent.config.yaml
    
    Returns:
        Dict containing experiment tracking configuration with defaults
    """
    # Determine project root (parent of utils directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config_path = os.path.join(parent_dir, "mcp_agent.config.yaml")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("experiment_tracking", {})
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load experiment config: {e}")
    
    # Default fallback configuration
    return {
        "enabled": False,
        "provider": "console"
    }


class AbstractExperimentTracker(ABC):
    """
    An abstract base class that defines the contract for all experiment trackers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the tracker with the given configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def trace(self, func: Callable) -> Callable:
        """A method that can act as a decorator for tracing function calls."""
        pass
    
    @abstractmethod
    def log_metrics(self, data_dict: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to the tracking backend."""
        pass
    
    @abstractmethod
    def save_artifact(self, **kwargs) -> None:
        """Save an artifact to the tracking backend."""
        pass


class ConsoleExperimentTracker(AbstractExperimentTracker):
    """Console-based experiment tracker for development and debugging."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("Initialized Console Experiment Tracker")
    
    def trace(self, func: Callable) -> Callable:
        """Pass-through decorator for console tracking."""
        return func
    
    def log_metrics(self, data_dict: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to console."""
        step_info = f" (step {step})" if step is not None else ""
        self.logger.info(f"Metrics{step_info}: {data_dict}")
    
    def save_artifact(self, **kwargs) -> None:
        """Log artifact save to console."""
        self.logger.info(f"Artifact saved: {kwargs}")


class WandBWeaveExperimentTracker(AbstractExperimentTracker):
    """Combined W&B and Weave experiment tracker."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.run_id: Optional[str] = None
        self.wandb_config: Dict[str, Any] = {}
        
        # Initialize Weave
        self.weave_project = config.get("weave", {}).get("project", "deepcode-experiments")
        initialize_weave(project_name=self.weave_project)
        
        # Initialize W&B only if reuired by config
        self.wandb_config = config.get("wandb", {})
        if len(self.wandb_config) > 0:
            self.run_id = initialize_wandb_run(

                project=self.wandb_config.get("project", "deepcode-experiments"),
                entity=self.wandb_config.get("entity"),
                stage="implementation"
            )
            
            if self.run_id is None:
                raise RuntimeError("Failed to initialize W&B run")
            
            self.logger.info("Initialized W&B/Weave Experiment Tracker")
    
    def trace(self, func: Callable) -> Callable:
        """Apply the @weave.op() decorator."""
        return weave.op()(func)
    
    def log_metrics(self, data_dict: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        wandb.log(data_dict, step=step)
    
    def save_artifact(self, **kwargs) -> None:
        """Save an artifact by calling the imported utility function."""
        upload_artifact(**kwargs)


def get_experiment_tracker() -> AbstractExperimentTracker:
    """
    Factory function to get the appropriate experiment tracker based on configuration.
    
    Returns:
        An instance of the appropriate experiment tracker
    """
    config = load_experiment_config()
    print(f"Experiment Tracking Config: {config}")
    
    if not config.get("enabled", False) or config.get("provider", "console") == "console":
        return ConsoleExperimentTracker(config)
    
    provider = config.get("provider", "console")
    
    if provider == "wandb_weave":
        tracker = WandBWeaveExperimentTracker(config)
        return tracker
    else:
        logging.getLogger(__name__).warning(f"Unknown provider '{provider}', using console")
        return ConsoleExperimentTracker(config)

