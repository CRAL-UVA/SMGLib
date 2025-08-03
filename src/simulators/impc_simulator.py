"""
IMPC-DR (Integrated Model Predictive Control with Deadlock Resolution) simulator.
"""

import sys
import os
from pathlib import Path

# Import from the original run_simulation.py for now (maintains exact functionality)
sys.path.append(str(Path(__file__).parent.parent.parent))
import run_simulation

def run_impc_simulation(num_robots: int, env_type: str):
    """
    Run Social-IMPC-DR simulation.
    
    Args:
        num_robots: Number of robots to simulate
        env_type: Environment type ('hallway', 'doorway', 'intersection')
    
    Returns:
        dict: Simulation results including makespan, flow_rate, completion_data
    """
    try:
        # Use the original run_social_impc_dr function
        return run_simulation.run_social_impc_dr(num_robots, env_type)
    except Exception as e:
        print(f"IMPC-DR simulation error: {e}")
        return None 