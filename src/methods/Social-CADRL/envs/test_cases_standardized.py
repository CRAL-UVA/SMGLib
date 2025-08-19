"""
Standardized test cases for Social-CADRL using centralized environment configuration.
This module provides consistent environment layouts and agent positions across all methods.
"""

import numpy as np
import sys
from pathlib import Path

# Import standardized environment configuration
sys.path.append(str(Path(__file__).resolve().parents[4] / 'src'))
from utils import StandardizedEnvironment

from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.agent import Agent
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor

def get_testcase_standardized_doorway(policies=["learning", "GA3C_CADRL", "CADRL"]):
    """
    Get standardized doorway test case using centralized environment configuration.
    """
    # Get standardized agent positions
    positions = StandardizedEnvironment.get_standard_agent_positions('doorway', 2)
    
    agents = []
    for i, pos in enumerate(positions):
        agent = Agent(
            pos['start'][0],  # start_x
            pos['start'][1],  # start_y
            pos['goal'][0],   # goal_x
            pos['goal'][1],   # goal_y
            StandardizedEnvironment.DEFAULT_AGENT_RADIUS,
            StandardizedEnvironment.DEFAULT_PREF_SPEED,
            0.0,  # heading (will be calculated automatically)
            CADRLPolicy,
            UnicycleDynamics,
            [OtherAgentsStatesSensor],
            i,
        )
        agents.append(agent)
    
    # Add obstacle agents (wall)
    obstacles = StandardizedEnvironment.get_doorway_obstacles()
    for i, obs_pos in enumerate(obstacles):
        obstacle_agent = Agent(
            obs_pos[0],  # start_x
            obs_pos[1],  # start_y
            obs_pos[0],  # goal_x (same as start for stationary obstacles)
            obs_pos[1],  # goal_y (same as start for stationary obstacles)
            StandardizedEnvironment.DEFAULT_AGENT_RADIUS,
            0.0,  # pref_speed (0 for stationary obstacles)
            0.0,  # heading
            CADRLPolicy,
            UnicycleDynamics,
            [OtherAgentsStatesSensor],
            len(agents) + i,
        )
        agents.append(obstacle_agent)
    
    return agents

def get_testcase_standardized_hallway(policies=["learning", "GA3C_CADRL", "CADRL"]):
    """
    Get standardized hallway test case using centralized environment configuration.
    """
    # Get standardized agent positions
    positions = StandardizedEnvironment.get_standard_agent_positions('hallway', 2)
    
    agents = []
    for i, pos in enumerate(positions):
        agent = Agent(
            pos['start'][0],  # start_x
            pos['start'][1],  # start_y
            pos['goal'][0],   # goal_x
            pos['goal'][1],   # goal_y
            StandardizedEnvironment.DEFAULT_AGENT_RADIUS,
            StandardizedEnvironment.DEFAULT_PREF_SPEED,
            0.0,  # heading (will be calculated automatically)
            CADRLPolicy,
            UnicycleDynamics,
            [OtherAgentsStatesSensor],
            i,
        )
        agents.append(agent)
    
    # Add obstacle agents (walls)
    obstacles = StandardizedEnvironment.get_hallway_obstacles()
    for i, obs_pos in enumerate(obstacles):
        obstacle_agent = Agent(
            obs_pos[0],  # start_x
            obs_pos[1],  # start_y
            obs_pos[0],  # goal_x (same as start for stationary obstacles)
            obs_pos[1],  # goal_y (same as start for stationary obstacles)
            StandardizedEnvironment.DEFAULT_AGENT_RADIUS,
            0.0,  # pref_speed (0 for stationary obstacles)
            0.0,  # heading
            CADRLPolicy,
            UnicycleDynamics,
            [OtherAgentsStatesSensor],
            len(agents) + i,
        )
        agents.append(obstacle_agent)
    
    return agents

def get_testcase_standardized_intersection(policies=["learning", "GA3C_CADRL", "CADRL"]):
    """
    Get standardized intersection test case using centralized environment configuration.
    """
    # Get standardized agent positions
    positions = StandardizedEnvironment.get_standard_agent_positions('intersection', 2)
    
    agents = []
    for i, pos in enumerate(positions):
        agent = Agent(
            pos['start'][0],  # start_x
            pos['start'][1],  # start_y
            pos['goal'][0],   # goal_x
            pos['goal'][1],   # goal_y
            StandardizedEnvironment.DEFAULT_AGENT_RADIUS,
            StandardizedEnvironment.DEFAULT_PREF_SPEED,
            0.0,  # heading (will be calculated automatically)
            CADRLPolicy,
            UnicycleDynamics,
            [OtherAgentsStatesSensor],
            i,
        )
        agents.append(agent)
    
    # Add obstacle agents (walls)
    obstacles = StandardizedEnvironment.get_intersection_obstacles()
    for i, obs_pos in enumerate(obstacles):
        obstacle_agent = Agent(
            obs_pos[0],  # start_x
            obs_pos[1],  # start_y
            obs_pos[0],  # goal_x (same as start for stationary obstacles)
            obs_pos[1],  # goal_y (same as start for stationary obstacles)
            StandardizedEnvironment.DEFAULT_AGENT_RADIUS,
            0.0,  # pref_speed (0 for stationary obstacles)
            0.0,  # heading
            CADRLPolicy,
            UnicycleDynamics,
            [OtherAgentsStatesSensor],
            len(agents) + i,
        )
        agents.append(obstacle_agent)
    
    return agents

def get_testcase_standardized_random(env_type='doorway', num_agents=2, policies=["learning", "GA3C_CADRL", "CADRL"]):
    """
    Get standardized random test case for the specified environment type.
    """
    if env_type == 'doorway':
        return get_testcase_standardized_doorway(policies)
    elif env_type == 'hallway':
        return get_testcase_standardized_hallway(policies)
    elif env_type == 'intersection':
        return get_testcase_standardized_intersection(policies)
    else:
        # Fallback to doorway
        return get_testcase_standardized_doorway(policies)

# Convenience function for backward compatibility
def get_testcase_standardized(env_type='doorway', num_agents=2):
    """Get standardized test case for the specified environment type."""
    return get_testcase_standardized_random(env_type, num_agents) 