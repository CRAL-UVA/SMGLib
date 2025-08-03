"""
Configuration management for SMGLib simulations.
"""

from pathlib import Path
from typing import Dict, Any, List
import json

class SimulationConfig:
    """Configuration class for simulation parameters."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.methods_dir = self.base_dir / "src" / "methods"
        
        # Default simulation parameters
        self.default_params = {
            "num_robots": 2,
            "env_type": "hallway",  # hallway, doorway, intersection
            "time_limit": 30.0,
            "time_step": 0.1,
            "animation_fps": 5,
            "save_animations": True,
            "save_trajectories": True
        }
    
    def get_method_dir(self, method_name: str) -> Path:
        """Get the directory path for a specific method."""
        method_map = {
            "orca": "Social-ORCA",
            "cadrl": "Social-CADRL", 
            "impc": "Social-IMPC-DR",
            "Social-ORCA": "Social-ORCA",
            "Social-CADRL": "Social-CADRL",
            "Social-IMPC-DR": "Social-IMPC-DR"
        }
        
        method_dir_name = method_map.get(method_name, method_name)
        return self.methods_dir / method_dir_name
    
    def get_configs_dir(self, method_name: str) -> Path:
        """Get the configs directory for a method."""
        return self.get_method_dir(method_name) / "configs"
    
    def get_logs_dir(self, method_name: str) -> Path:
        """Get the logs directory for a method."""
        return self.get_method_dir(method_name) / "logs"
    
    def create_directories(self, method_name: str):
        """Create necessary directories for a method."""
        configs_dir = self.get_configs_dir(method_name)
        logs_dir = self.get_logs_dir(method_name)
        
        configs_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create trajectories subdirectory
        (logs_dir / "trajectories").mkdir(exist_ok=True)
        (logs_dir / "animations").mkdir(exist_ok=True)
    
    def get_param(self, key: str, default=None):
        """Get a configuration parameter."""
        return self.default_params.get(key, default)
    
    def set_param(self, key: str, value: Any):
        """Set a configuration parameter."""
        self.default_params[key] = value
    
    def load_from_file(self, config_file: Path):
        """Load configuration from a JSON file."""
        if config_file.exists():
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                self.default_params.update(loaded_config)
    
    def save_to_file(self, config_file: Path):
        """Save configuration to a JSON file."""
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(self.default_params, f, indent=2)

# Global configuration instance
config = SimulationConfig() 