"""
ORCA (Optimal Reciprocal Collision Avoidance) simulator.
"""

import sys
import os
from pathlib import Path

# Import from the original run_simulation.py for now (maintains exact functionality)
sys.path.append(str(Path(__file__).parent.parent.parent))
import run_simulation

def run_orca_simulation(num_robots: int, env_type: str):
    """
    Run Social-ORCA simulation.
    
    Args:
        num_robots: Number of robots to simulate
        env_type: Environment type ('hallway', 'doorway', 'intersection')
    
    Returns:
        dict: Simulation results including makespan, flow_rate, completion_data
    """
    try:
        # Use the original run_social_orca function
        return run_simulation.run_social_orca(num_robots, env_type)
    except Exception as e:
        print(f"ORCA simulation error: {e}")
        return None 