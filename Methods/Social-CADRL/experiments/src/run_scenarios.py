# Add parent directory to Python path for imports
import os
import sys

# Get the absolute path to the Social-CADRL envs directory
current_dir = os.path.dirname(os.path.abspath(__file__))
envs_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "envs")
sys.path.append(envs_dir)

import gym
import numpy as np
import argparse
import json
from gym import spaces
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent freezing
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
import pandas as pd
import csv
import os
from pathlib import Path

# Disable observation space checking
gym.logger.set_level(40)
os.environ["GYM_CONFIG_CLASS"] = "Example"
os.environ["GYM_DISABLE_OBSERVATION_SPACE_CHECK"] = "1"

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.agent import Agent
from envs.dynamics.UnicycleDynamics import UnicycleDynamics
from envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor
from envs.policies.CADRLPolicy import CADRLPolicy

def parse_agent_args(agent_str):
    """Parse agent string in format 'start_x,start_y:goal_x,goal_y'"""
    try:
        start, goal = agent_str.split(':')
        start_x, start_y = map(float, start.split(','))
        goal_x, goal_y = map(float, goal.split(','))
        return start_x, start_y, goal_x, goal_y
    except:
        print(f"Error: Invalid agent format. Expected 'start_x,start_y:goal_x,goal_y', got '{agent_str}'")
        sys.exit(1)

def create_base_intersection_agents():
    """Create the base agents for the intersection scenario"""
    agents = []
    # Add static agents around the intersection
    positions = [
        (-3.0, 2.0), (-2.0, 2.0), (-1.0, 2.0), (-1.0, 3.0), (-1.0, 4.0), (-1.0, 5.0),
        (3.0, 2.0), (4.0, 2.0), (5.0, 2.0), (3.0, 3.0), (3.0, 4.0), (3.0, 5.0),
        (-3.0, -2.0), (-2.0, -2.0), (-1.0, -2.0), (-1.0, -3.0), (-1.0, -4.0), (-1.0, -5.0),
        (3.0, -2.0), (4.0, -2.0), (5.0, -2.0), (3.0, -3.0), (3.0, -4.0), (3.0, -5.0)
    ]
    
    for i, (x, y) in enumerate(positions):
        agents.append(Agent(
            x, y, x, y,  # Static agents have same start and goal
            0.5, 1.0, np.pi,
            CADRLPolicy,  # Pass the class, not an instance
            UnicycleDynamics,
            [OtherAgentsStatesSensor],
            i
        ))
    return agents

def create_base_hallway_agents():
    """Create the base agents for the hallway scenario"""
    agents = []
    # Add static agents along the hallway walls
    positions = [
        (-3.0, 2.0), (-2.0, 2.0), (-1.0, 2.0), (0.0, 2.0), (1.0, 2.0), (2.0, 2.0), (3.0, 2.0),
        (-3.0, -2.0), (-2.0, -2.0), (-1.0, -2.0), (0.0, -2.0), (1.0, -2.0), (2.0, -2.0), (3.0, -2.0)
    ]
    
    for i, (x, y) in enumerate(positions):
        agents.append(Agent(
            x, y, x, y,  # Static agents have same start and goal
            0.5, 1.0, np.pi,
            CADRLPolicy,  # Pass the class, not an instance
            UnicycleDynamics,
            [OtherAgentsStatesSensor],
            i
        ))
    return agents

def create_base_doorway_agents():
    """Create the base agents for the doorway scenario (vertical configuration)"""
    agents = []
    # Add static agents forming vertical wall with doorway gap
    # Single vertical line at x=0.0 with gap in the middle
    positions = []
    
    # Bottom side of vertical wall: y from -9.0 to -2.0
    for y in [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0]:
        positions.append((0.0, y))
    
    # Top side of vertical wall: y from 2.0 to 9.0  
    for y in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
        positions.append((0.0, y))
    
    for i, (x, y) in enumerate(positions):
        agents.append(Agent(
            x, y, x, y,  # Static agents have same start and goal
            0.5, 1.0, np.pi,
            CADRLPolicy,  # Pass the class, not an instance
            UnicycleDynamics,
            [OtherAgentsStatesSensor],
            i
        ))
    return agents

def get_obstacle_positions(scenario_type):
    """Get obstacle positions for each scenario"""
    if scenario_type == 'intersection':
        return [
            (-3.0, 2.0), (-2.0, 2.0), (-1.0, 2.0), (-1.0, 3.0), (-1.0, 4.0), (-1.0, 5.0),
            (3.0, 2.0), (4.0, 2.0), (5.0, 2.0), (3.0, 3.0), (3.0, 4.0), (3.0, 5.0),
            (-3.0, -2.0), (-2.0, -2.0), (-1.0, -2.0), (-1.0, -3.0), (-1.0, -4.0), (-1.0, -5.0),
            (3.0, -2.0), (4.0, -2.0), (5.0, -2.0), (3.0, -3.0), (3.0, -4.0), (3.0, -5.0)
        ]
    elif scenario_type == 'hallway':
        return [
            (-3.0, 2.0), (-2.0, 2.0), (-1.0, 2.0), (0.0, 2.0), (1.0, 2.0), (2.0, 2.0), (3.0, 2.0),
            (-3.0, -2.0), (-2.0, -2.0), (-1.0, -2.0), (0.0, -2.0), (1.0, -2.0), (2.0, -2.0), (3.0, -2.0)
        ]
    elif scenario_type == 'doorway':
        obstacles = []
        # Create vertical wall
        # Bottom side of vertical wall
        for y in [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0]:
            obstacles.append((0.0, y))
        # Top side of vertical wall
        for y in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
            obstacles.append((0.0, y))
        return obstacles
    return []

def validate_agent_position(x, y, scenario_type, position_type="position"):
    """Validate that an agent position doesn't conflict with obstacles"""
    obstacles = get_obstacle_positions(scenario_type)
    
    # Check for exact overlap with obstacles
    for obs_x, obs_y in obstacles:
        distance = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
        if distance < 0.8:  # Agent radius + obstacle radius
            return False, f"Agent {position_type} ({x:.1f}, {y:.1f}) is too close to obstacle at ({obs_x:.1f}, {obs_y:.1f}). Minimum distance is 0.8."
    
    # Check scenario-specific bounds
    if scenario_type == 'intersection':
        if abs(x) > 8 or abs(y) > 8:
            return False, f"Agent {position_type} ({x:.1f}, {y:.1f}) is outside intersection bounds (±8, ±8)."
    elif scenario_type == 'hallway':
        if abs(y) > 1.5:
            return False, f"Agent {position_type} ({x:.1f}, {y:.1f}) is outside hallway bounds (y must be within ±1.5)."
        if abs(x) > 6:
            return False, f"Agent {position_type} ({x:.1f}, {y:.1f}) is outside hallway bounds (x must be within ±6)."
    elif scenario_type == 'doorway':
        if abs(x) > 0.8:
            return False, f"Agent {position_type} ({x:.1f}, {y:.1f}) is outside doorway opening (x must be within ±0.8)."
        if abs(y) > 12:
            return False, f"Agent {position_type} ({x:.1f}, {y:.1f}) is outside scenario bounds (y must be within ±12)."
    
    return True, ""

def get_default_values(scenario_type):
    """Get default values for this scenario"""
    defaults = {}
    
    if scenario_type == 'intersection':
        # Default values for first dynamic agent
        defaults['start_x'] = -3.0
        defaults['start_y'] = 0.0
        defaults['goal_x'] = 5.0
        defaults['goal_y'] = 0.0
        defaults['radius'] = 0.5
        defaults['pref_speed'] = 1.0
        defaults['heading'] = 0.0
        
        # Second dynamic agent defaults (different from first)
        defaults['agent2'] = {
            'start_x': 2.0,
            'start_y': -5.0,
            'goal_x': 5.0,
            'goal_y': 0.0,
            'radius': 0.5,
            'pref_speed': 1.0,
            'heading': np.pi
        }
    elif scenario_type == 'hallway':
        # First agent: left to right
        defaults['start_x'] = -4.0
        defaults['start_y'] = 0.0
        defaults['goal_x'] = 4.0
        defaults['goal_y'] = 0.0
        defaults['radius'] = 0.5
        defaults['pref_speed'] = 1.0
        defaults['heading'] = 0.0
        
        # Second agent: right to left
        defaults['agent2'] = {
            'start_x': 4.0,
            'start_y': 0.0,
            'goal_x': -4.0,
            'goal_y': 0.0,
            'radius': 0.5,
            'pref_speed': 1.0,
            'heading': np.pi
        }
    elif scenario_type == 'doorway':
        # First agent: left side going through doorway to right side
        defaults['start_x'] = -3.0
        defaults['start_y'] = 0.0
        defaults['goal_x'] = 3.0
        defaults['goal_y'] = 0.0
        defaults['radius'] = 0.5
        defaults['pref_speed'] = 1.0
        defaults['heading'] = 0.0
        
        # Second agent: right side going through doorway to left side
        defaults['agent2'] = {
            'start_x': 3.0,
            'start_y': 0.0,
            'goal_x': -3.0,
            'goal_y': 0.0,
            'radius': 0.5,
            'pref_speed': 1.0,
            'heading': np.pi
        }
    
    return defaults

def get_float_input(prompt, default=None):
    """Get float input from user with default value"""
    while True:
        try:
            if default is not None:
                value = input(f"{prompt} (default: {default}): ").strip()
                if not value:
                    return default
                return float(value)
            else:
                return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_scenario_choice():
    """Get scenario choice from user"""
    while True:
        print("\nAvailable Scenarios:")
        print("1. Intersection")
        print("2. Hallway")
        print("3. Doorway")
        try:
            choice = int(input("\nEnter scenario number (1-3): "))
            if 1 <= choice <= 3:
                scenarios = {1: 'intersection', 2: 'hallway', 3: 'doorway'}
                return scenarios[choice]
            else:
                print("Invalid choice. Please enter a number between 1 and 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_num_agents():
    """Get number of agents from user"""
    while True:
        try:
            num = int(input("\nEnter number of agents to add (0 for default scenario): "))
            if num >= 0:
                return num
            else:
                print("Please enter a non-negative number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_agent_parameters(num_agents, scenario_type):
    """Get agent parameters from user"""
    user_agents = []
    defaults = get_default_values(scenario_type)
    
    # Print scenario-specific instructions
    if scenario_type == 'intersection':
        print("\nIntersection Configuration:")
        print("- The intersection has walls at x=30-31 and y=30-31")
        print("- X and Y coordinates should be between -10 and 10")
    elif scenario_type == 'hallway':
        print("\nHallway Configuration:")
        print("- The hallway has walls at y=±2")
        print("- Navigable space has y between ±1.5")
        print("- X coordinates should be between -6 and 6")
    elif scenario_type == 'doorway':
        print("\nDoorway Configuration:")
        print("- The doorway has a vertical wall with a narrow opening")
        print("- Navigable space has x between ±0.8")
        print("- Y coordinates should be between -12 and 12")
    
    for i in range(num_agents):
        print(f"\n--- Agent {i+1} Parameters ---")
        
        # Use defaults for first two agents if available
        if i == 0:
            start_x = get_float_input("Start X position", defaults['start_x'])
            start_y = get_float_input("Start Y position", defaults['start_y'])
            goal_x = get_float_input("Goal X position", defaults['goal_x'])
            goal_y = get_float_input("Goal Y position", defaults['goal_y'])
            radius = get_float_input("Agent radius", defaults['radius'])
            pref_speed = get_float_input("Preferred speed", defaults['pref_speed'])
            heading = get_float_input("Initial heading (radians)", defaults['heading'])
        elif i == 1 and 'agent2' in defaults:
            agent2_defaults = defaults['agent2']
            start_x = get_float_input("Start X position", agent2_defaults['start_x'])
            start_y = get_float_input("Start Y position", agent2_defaults['start_y'])
            goal_x = get_float_input("Goal X position", agent2_defaults['goal_x'])
            goal_y = get_float_input("Goal Y position", agent2_defaults['goal_y'])
            radius = get_float_input("Agent radius", agent2_defaults['radius'])
            pref_speed = get_float_input("Preferred speed", agent2_defaults['pref_speed'])
            heading = get_float_input("Initial heading (radians)", agent2_defaults['heading'])
        else:
            start_x = get_float_input("Start X position")
            start_y = get_float_input("Start Y position")
            goal_x = get_float_input("Goal X position")
            goal_y = get_float_input("Goal Y position")
            radius = get_float_input("Agent radius", 0.5)
            pref_speed = get_float_input("Preferred speed", 1.0)
            heading = get_float_input("Initial heading (radians)", 0.0)
        
        user_agents.append((start_x, start_y, goal_x, goal_y, radius, pref_speed, heading))
        print(f"Agent {i+1} configured: Start=({start_x:.1f}, {start_y:.1f}), Goal=({goal_x:.1f}, {goal_y:.1f})")
    
    return user_agents

def run_scenario(scenario_type, user_agents, num_steps=150):
    """Run the specified scenario with user-defined agents"""
    # Create single tf session for all experiments
    import tensorflow.compat.v1 as tf
    tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()

    # Import necessary modules
    from gym_collision_avoidance.envs import Config
    from gym_collision_avoidance.envs import test_cases as tc
    from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
    from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor
    from gym_collision_avoidance.envs.agent import Agent

    # Initialize agents based on scenario - create only static obstacles
    if scenario_type == 'intersection':
        agents = create_base_intersection_agents()
        obstacle_coords = [
            (-3.0, 2.0), (-2.0, 2.0), (-1.0, 2.0), (-1.0, 3.0), (-1.0, 4.0), (-1.0, 5.0),
            (3.0, 2.0), (4.0, 2.0), (5.0, 2.0), (3.0, 3.0), (3.0, 4.0), (3.0, 5.0),
            (-3.0, -2.0), (-2.0, -2.0), (-1.0, -2.0), (-1.0, -3.0), (-1.0, -4.0), (-1.0, -5.0),
            (3.0, -2.0), (4.0, -2.0), (5.0, -2.0), (3.0, -3.0), (3.0, -4.0), (3.0, -5.0)
        ]
    elif scenario_type == 'hallway':
        agents = create_base_hallway_agents()
        obstacle_coords = [
            (-3.0, 2.0), (-2.0, 2.0), (-1.0, 2.0), (0.0, 2.0), (1.0, 2.0), (2.0, 2.0), (3.0, 2.0),
            (-3.0, -2.0), (-2.0, -2.0), (-1.0, -2.0), (0.0, -2.0), (1.0, -2.0), (2.0, -2.0), (3.0, -2.0)
        ]
    elif scenario_type == 'doorway':
        agents = create_base_doorway_agents()
        obstacle_coords = []
        # Create vertical wall for visualization
        # Bottom side of vertical wall
        for y in [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0]:
            obstacle_coords.append((0.0, y))
        # Top side of vertical wall
        for y in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
            obstacle_coords.append((0.0, y))
    else:
        agents = []
        obstacle_coords = []

    # All base agents are static obstacles (start == goal)
    dynamic_agents = []
    print(f"Base scenario has {len(agents)} static obstacle agents")

    # Add user-defined agents if provided (these will be the ONLY dynamic agents)
    if user_agents:
        print(f"\nCreating {len(user_agents)} user agents...")
        for i, (start_x, start_y, goal_x, goal_y, radius, pref_speed, heading) in enumerate(user_agents):
            print(f"Agent {i+1}: Start=({start_x:.1f}, {start_y:.1f}), Goal=({goal_x:.1f}, {goal_y:.1f})")
            
            new_agent = Agent(
                start_x, start_y, goal_x, goal_y,
                radius, pref_speed, heading,
                tc.policy_dict["CADRL"],
                UnicycleDynamics,
                [OtherAgentsStatesSensor],
                len(agents)
            )
            
            agents.append(new_agent)
            dynamic_agents.append(new_agent)

    # Set config for the number of agents
    Config.MAX_NUM_AGENTS_IN_ENVIRONMENT = len(agents)
    
    print(f"Total agents: {len(agents)}, Dynamic agents: {len(dynamic_agents)}")

    # Create the environment
    env = gym.make("CollisionAvoidance-v0")
    
    # Disable built-in animation to avoid ffmpeg issues - we'll create our own
    if hasattr(Config, 'ANIMATE_EPISODES'):
        Config.ANIMATE_EPISODES = False

    # Initialize networks for agents that need it
    print("Initializing CADRL networks...")
    for i, agent in enumerate(agents):
        if hasattr(agent.policy, "initialize_network"):
            try:
                agent.policy.initialize_network()
            except Exception as e:
                print(f"Warning: Could not initialize network for agent {i}: {e}")
    
    # Set agents in environment
    env.set_agents(agents)
    
    # Get initial observations
    obs = env.reset()
    
    # Store positions for animation and evaluation
    positions_history = []
    
    # Initialize data collection for evaluation metrics (only for dynamic/moving agents)
    agent_tracking_data = []
    dynamic_agent_counter = 0
    for j, agent in enumerate(dynamic_agents):
        agent_idx = agents.index(agent)
        start_pos = [agent.pos_global_frame[0], agent.pos_global_frame[1]]
        goal_pos = [agent.goal_global_frame[0], agent.goal_global_frame[1]]
        
        # Only track agents that are actually moving (have different start and goal)
        distance_start_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(start_pos))
        if distance_start_to_goal > 0.1:  # Threshold to identify moving robots
            agent_data = {
                'id': dynamic_agent_counter,  # Use 0-based counter for dynamic agents
                'original_id': agent_idx,  # Keep original ID for reference
                'positions': [],
                'velocities': [],
                'start_pos': start_pos,
                'goal_pos': goal_pos
            }
            agent_tracking_data.append(agent_data)
            dynamic_agent_counter += 1
    
    print(f"Tracking {len(agent_tracking_data)} moving agents for evaluation metrics")
    
    print(f"\nStarting simulation for {num_steps} steps...")
    time_step = 0.1  # CADRL simulation time step
    
    # Store completion times for makespan ratio
    completion_times = {}
    
    for i in range(num_steps):
        # Get actions for dynamic agents only
        actions = {}
        
        for j, agent in enumerate(dynamic_agents):
            agent_idx = agents.index(agent)
            
            # Get the agent's observation
            if isinstance(obs, list):
                agent_obs = obs[agent_idx] if agent_idx < len(obs) else None
            else:
                agent_obs = obs
            
            try:
                if hasattr(agent.policy, 'find_next_action') and agent_obs is not None:
                    # CADRL policy requires (obs, agents, agent_index) parameters
                    action = agent.policy.find_next_action(agent_obs, agents, agent_idx)
                    actions[agent_idx] = action
                    
                    # Debug output for first few steps
                    if i < 3:
                        print(f"Step {i}, Agent {j+1}: Action={action}, Pos=({agent.pos_global_frame[0]:.2f}, {agent.pos_global_frame[1]:.2f})")
                else:
                    # Fallback: direct movement toward goal
                    direction = agent.goal_global_frame - agent.pos_global_frame
                    distance = np.linalg.norm(direction)
                    if distance > 0.1:
                        direction = direction / distance
                        action = direction * agent.pref_speed
                        actions[agent_idx] = action
                        if i < 3:
                            print(f"Step {i}, Agent {j+1}: Fallback action={action}")
                    else:
                        actions[agent_idx] = np.array([0.0, 0.0])
                        
            except Exception as e:
                print(f"Error getting action for agent {j+1}: {e}")
                # Emergency fallback
                direction = agent.goal_global_frame - agent.pos_global_frame
                distance = np.linalg.norm(direction)
                if distance > 0.1:
                    direction = direction / distance
                    action = direction * agent.pref_speed
                    actions[agent_idx] = action
        
        # Execute simulation step
        obs, rewards, terminated, truncated, which_agents_done = env.step(actions)
        
        # Store current positions of all agents
        current_positions = []
        for agent in agents:
            current_positions.append([agent.pos_global_frame[0], agent.pos_global_frame[1]])
        positions_history.append(current_positions)
        
        # Collect trajectory and velocity data for evaluation (only for tracked moving agents)
        for tracking_idx, tracking_data in enumerate(agent_tracking_data):
            # Find the corresponding dynamic agent by original ID
            agent_idx = tracking_data['original_id']
            
            # Find the agent object in dynamic_agents
            agent = None
            for dyn_agent in dynamic_agents:
                if agents.index(dyn_agent) == agent_idx:
                    agent = dyn_agent
                    break
            
            if agent is not None:
                current_pos = [agent.pos_global_frame[0], agent.pos_global_frame[1]]
                
                # Store position
                agent_tracking_data[tracking_idx]['positions'].append(current_pos)
                
                # Calculate and store velocity
                if len(agent_tracking_data[tracking_idx]['positions']) > 1:
                    prev_pos = agent_tracking_data[tracking_idx]['positions'][-2]
                    velocity = [
                        (current_pos[0] - prev_pos[0]) / time_step,
                        (current_pos[1] - prev_pos[1]) / time_step
                    ]
                else:
                    # First step, use action as velocity estimate
                    velocity = actions.get(agent_idx, [0.0, 0.0])
                
                agent_tracking_data[tracking_idx]['velocities'].append(velocity)
                
                # Check if agent has reached goal and record completion time
                distance_to_goal = np.linalg.norm(agent.goal_global_frame - agent.pos_global_frame)
                if distance_to_goal <= 0.3 and tracking_data['id'] not in completion_times:
                    completion_times[tracking_data['id']] = i * time_step
                    print(f"Agent {tracking_data['id']} reached goal at time {completion_times[tracking_data['id']]:.2f}s")
        
        # Check if any dynamic agents have reached their goals
        agents_at_goal = []
        agents_still_moving = []
        for j, agent in enumerate(dynamic_agents):
            distance_to_goal = np.linalg.norm(agent.goal_global_frame - agent.pos_global_frame)
            if distance_to_goal <= 0.3:  # More lenient goal tolerance
                agents_at_goal.append(agent)
            else:
                agents_still_moving.append(agent)
        
        all_dynamic_done = len(agents_still_moving) == 0
        
        if all_dynamic_done and i > 10:  # Ensure some minimum simulation time
            print(f"All dynamic agents reached their goals at step {i}!")
            # Continue for a few more steps to show the final state
            extra_steps = min(20, num_steps - i - 1)
            for extra_i in range(extra_steps):
                # Store the same positions for pause effect
                current_positions = []
                for agent in agents:
                    current_positions.append([agent.pos_global_frame[0], agent.pos_global_frame[1]])
                positions_history.append(current_positions)
            print(f"Added {extra_steps} pause frames after completion")
            break
        
        # Print progress every 10 steps
        if i % 10 == 0 and i > 0:
            print(f"Step {i}: {len(agents_still_moving)} agents still moving...")

    # Reset environment at the end
    env.reset()
    
    # Calculate and display makespan ratio
    if len(completion_times) >= 2:
        fastest_time = min(completion_times.values())
        if fastest_time > 0:
            makespan_ratios = {agent_idx: time/fastest_time for agent_idx, time in completion_times.items()}
            avg_makespan_ratio = sum(makespan_ratios.values()) / len(makespan_ratios)
            max_makespan_ratio = max(makespan_ratios.values())
            
            print("\nMakespan Ratio Metrics:")
            print("=" * 65)
            print(f"Average Makespan Ratio: {avg_makespan_ratio:.4f}")
            print(f"Maximum Makespan Ratio: {max_makespan_ratio:.4f}")
            print(f"Fastest Agent Time: {fastest_time:.4f} seconds")
            for agent_idx, ratio in makespan_ratios.items():
                print(f"Agent {agent_idx} Makespan Ratio: {ratio:.4f} (Time: {completion_times[agent_idx]:.4f}s)")
            print("=" * 65)

    # Evaluate CADRL performance with comprehensive metrics
    evaluation_results = evaluate_cadrl_performance(agent_tracking_data, scenario_type, time_step)
    
    # Save trajectory data for further analysis
    output_dir = Path("logs/cadrl_evaluation")
    save_cadrl_trajectory_data(agent_tracking_data, output_dir, time_step)
    
    return evaluation_results

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Social Collision Avoidance Scenario Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_scenarios.py
  
  # Run intersection with 2 agents
  python run_scenarios.py --scenario intersection --agents 2 \\
    --agent1 "-3,0:5,0:0.5:1.0:0" --agent2 "2,-5:5,0:0.5:1.0:3.14"
  
  # Run hallway with 1 agent using defaults
  python run_scenarios.py --scenario hallway --agents 1
  
  # Load agents from JSON file
  python run_scenarios.py --scenario intersection --agents-file agents.json
  
Agent format: "start_x,start_y:goal_x,goal_y:radius:pref_speed:heading"
        """
    )
    
    parser.add_argument('--scenario', '-s', 
                       choices=['intersection', 'hallway', 'doorway'],
                       help='Scenario type to run')
    
    parser.add_argument('--agents', '-n', type=int,
                       help='Number of agents to create')
    
    parser.add_argument('--steps', type=int, default=150,
                       help='Maximum number of simulation steps (default: 150)')
    
    # Agent parameters
    for i in range(1, 11):  # Support up to 10 agents
        parser.add_argument(f'--agent{i}', 
                           help=f'Agent {i} parameters: "start_x,start_y:goal_x,goal_y:radius:pref_speed:heading"')
    
    parser.add_argument('--agents-file', 
                       help='JSON file containing agent configurations')
    
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for animations (default: ../animations)')
    
    parser.add_argument('--no-animation', action='store_true',
                       help='Skip animation generation')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')
    
    return parser.parse_args()

def parse_agent_string(agent_str, scenario_type, agent_num):
    """Parse agent parameter string"""
    try:
        parts = agent_str.split(':')
        if len(parts) < 2:
            raise ValueError("Missing goal position")
        
        # Parse start and goal positions
        start_parts = parts[0].split(',')
        goal_parts = parts[1].split(',')
        
        if len(start_parts) != 2 or len(goal_parts) != 2:
            raise ValueError("Start and goal must have x,y coordinates")
        
        start_x, start_y = map(float, start_parts)
        goal_x, goal_y = map(float, goal_parts)
        
        # Parse optional parameters with defaults
        defaults = get_default_values(scenario_type)
        
        if len(parts) > 2 and parts[2]:
            radius = float(parts[2])
        else:
            radius = defaults.get('radius', 0.5)
        
        if len(parts) > 3 and parts[3]:
            pref_speed = float(parts[3])
        else:
            pref_speed = defaults.get('pref_speed', 1.0)
        
        if len(parts) > 4 and parts[4]:
            heading = float(parts[4])
        else:
            # Use different default heading for second agent in intersection
            if scenario_type == 'intersection' and agent_num == 2 and 'agent2' in defaults:
                heading = defaults['agent2']['heading']
            else:
                heading = defaults.get('heading', 0.0)
        
        # Validate positions
        start_valid, start_msg = validate_agent_position(start_x, start_y, scenario_type, "start position")
        if not start_valid:
            raise ValueError(f"Agent {agent_num} {start_msg}")
        
        goal_valid, goal_msg = validate_agent_position(goal_x, goal_y, scenario_type, "goal position")
        if not goal_valid:
            raise ValueError(f"Agent {agent_num} {goal_msg}")
        
        return (start_x, start_y, goal_x, goal_y, radius, pref_speed, heading)
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid agent string '{agent_str}': {e}")

def load_agents_from_file(filename):
    """Load agent configurations from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        agents = []
        for agent_data in data.get('agents', []):
            agent = (
                agent_data['start_x'],
                agent_data['start_y'], 
                agent_data['goal_x'],
                agent_data['goal_y'],
                agent_data.get('radius', 0.5),
                agent_data.get('pref_speed', 1.0),
                agent_data.get('heading', 0.0)
            )
            agents.append(agent)
        
        return agents
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Error loading agents file '{filename}': {e}")

def create_default_agents(scenario_type, num_agents):
    """Create default agents for the scenario"""
    defaults = get_default_values(scenario_type)
    agents = []
    
    for i in range(num_agents):
        if i == 0:
            # First agent uses main defaults
            agent = (
                defaults['start_x'], defaults['start_y'],
                defaults['goal_x'], defaults['goal_y'],
                defaults['radius'], defaults['pref_speed'], defaults['heading']
            )
        elif i == 1 and 'agent2' in defaults:
            # Second agent uses agent2 defaults if available
            agent2 = defaults['agent2']
            agent = (
                agent2['start_x'], agent2['start_y'],
                agent2['goal_x'], agent2['goal_y'],
                agent2['radius'], agent2['pref_speed'], agent2['heading']
            )
        else:
            # Additional agents use variations of defaults
            if scenario_type == 'intersection':
                # Vary start positions for intersection
                start_positions = [(-4, 1), (1, -4), (-2, -1), (4, 2)]
                pos = start_positions[i % len(start_positions)]
                agent = (pos[0], pos[1], 5.0, 0.0, 0.5, 1.0, 0.0)
            elif scenario_type == 'hallway':
                # Vary start positions for hallway
                start_x = -3.0 + i * 0.5
                agent = (start_x, 0.0, 3.0, 0.0, 0.5, 1.0, 0.0)
            else:  # doorway
                # Vary start positions for doorway
                start_x = -5.0 + i * 0.3
                agent = (start_x, 0.0, 5.0, 0.0, 0.5, 1.0, 0.0)
        
        agents.append(agent)
    
    return agents

def calculate_nominal_path(start_pos, goal_pos, num_steps):
    """Calculate the nominal path (straight line) from start to goal."""
    x = np.linspace(start_pos[0], goal_pos[0], num_steps)
    y = np.linspace(start_pos[1], goal_pos[1], num_steps)
    return x, y

def calculate_path_deviation(actual_positions, start_pos, goal_pos):
    """Calculate path deviation metrics for a robot's trajectory."""
    if len(actual_positions) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Generate nominal path with same number of steps
    num_steps = len(actual_positions)
    nominal_x, nominal_y = calculate_nominal_path(start_pos, goal_pos, num_steps)
    
    # Calculate deviations at each step
    deviations = []
    for i, actual_pos in enumerate(actual_positions):
        nominal_pos = [nominal_x[i], nominal_y[i]]
        deviation = np.linalg.norm(np.array(actual_pos) - np.array(nominal_pos))
        deviations.append(deviation)
    
    # Calculate metrics
    avg_deviation = np.mean(deviations)
    max_deviation = np.max(deviations)
    total_path_length = 0.0
    
    # Calculate actual path length
    for i in range(1, len(actual_positions)):
        segment_length = np.linalg.norm(np.array(actual_positions[i]) - np.array(actual_positions[i-1]))
        total_path_length += segment_length
    
    # Calculate nominal path length
    nominal_path_length = np.linalg.norm(np.array(goal_pos) - np.array(start_pos))
    
    return avg_deviation, max_deviation, total_path_length, nominal_path_length

def calculate_average_delta_velocity(velocities):
    """Calculate average delta velocity for a robot's trajectory."""
    if len(velocities) < 2:
        return 0.0
    
    # Calculate resultant velocities
    resultant_velocities = []
    for vel in velocities:
        resultant = np.linalg.norm(vel)
        resultant_velocities.append(resultant)
    
    # Calculate differences between consecutive velocities
    velocity_diffs = []
    for i in range(1, len(resultant_velocities)):
        diff = abs(resultant_velocities[i] - resultant_velocities[i-1])
        velocity_diffs.append(diff)
    
    # Return average of absolute differences
    return np.mean(velocity_diffs) if velocity_diffs else 0.0

def get_gap_width(scenario_type):
    """Get the gap width (bottleneck width) for each scenario type."""
    if scenario_type == 'doorway':
        # Vertical doorway gap: from y=-2.0 to y=2.0
        return 4.0  # meters
    elif scenario_type == 'hallway':
        # Hallway navigable width: from y=-1.5 to y=1.5 (walls at ±2.0)
        return 3.0  # meters  
    elif scenario_type == 'intersection':
        # Intersection navigable width (approximate)
        return 6.0  # meters (conservative estimate)
    else:
        return 1.0  # default width for unknown scenarios

def calculate_flow_rate(agent_data, time_step, gap_width):
    """Calculate flow rate = number of moving agents / (make_span * gap_width)."""
    if not agent_data:
        return 0.0
    
    # Find completion times for each agent
    completion_times = []
    
    for agent in agent_data:
        positions = agent['positions']
        goal_pos = agent['goal_pos']
        
        # Find when agent reached goal (within 0.3 units)
        for i, pos in enumerate(positions):
            distance_to_goal = np.linalg.norm(np.array(pos) - np.array(goal_pos))
            if distance_to_goal <= 0.3:
                completion_time = i * time_step
                completion_times.append(completion_time)
                break
    
    if not completion_times:
        return 0.0
    
    # Make-span = maximum completion time (time for all agents to complete)
    make_span = max(completion_times)
    
    # Flow rate = number of moving agents / (make_span * gap_width)
    # Units: agents per second per meter
    if make_span > 0 and gap_width > 0:
        flow_rate = len(agent_data) / (make_span * gap_width)
    else:
        flow_rate = float('inf')  # Instantaneous completion or invalid gap width
    
    return flow_rate

def save_cadrl_trajectory_data(agent_data, output_dir, time_step):
    """Save CADRL trajectory data to CSV files for analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save velocity data
    velocity_csv = output_dir / "velocities.csv"
    with open(velocity_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['step']
        for agent in agent_data:
            header.extend([f'robot_{agent["id"]}_vx', f'robot_{agent["id"]}_vy'])
        writer.writerow(header)
        
        # Write velocity data
        max_steps = max(len(agent['velocities']) for agent in agent_data) if agent_data else 0
        for t in range(max_steps):
            row = [t]
            for agent in agent_data:
                if t < len(agent['velocities']):
                    vel = agent['velocities'][t]
                    row.extend([vel[0], vel[1]])
                else:
                    row.extend([0, 0])
            writer.writerow(row)
    
    # Save individual trajectory files
    for agent in agent_data:
        num_steps = len(agent['positions'])
        if num_steps == 0:
            continue
            
        nominal_x, nominal_y = calculate_nominal_path(agent['start_pos'], agent['goal_pos'], num_steps)
        
        output_csv = output_dir / f"robot_{agent['id']}_trajectory.csv"
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y', 'nominal_x', 'nominal_y'])
            
            for t in range(num_steps):
                pos = agent['positions'][t]
                writer.writerow([pos[0], pos[1], nominal_x[t], nominal_y[t]])
    
    return velocity_csv

def evaluate_cadrl_performance(agent_data, scenario_type, time_step=0.1):
    """Evaluate CADRL performance with comprehensive metrics for moving robots only."""
    print("\n" + "="*80)
    print("CADRL PERFORMANCE EVALUATION (MOVING ROBOTS ONLY)")
    print("="*80)
    
    if not agent_data:
        print("No moving agent data available for evaluation.")
        return
    
    print(f"Evaluating {len(agent_data)} moving robots")
    
    total_avg_deviation = 0.0
    total_max_deviation = 0.0
    total_avg_delta_velocity = 0.0
    
    # Calculate completion times for each agent based on velocity
    completion_times = []
    for agent in agent_data:
        agent_id = agent['id']
        velocities = agent['velocities']
        positions = agent['positions']
        goal_pos = agent['goal_pos']
        
        # Calculate resultant velocities
        resultant_velocities = [np.sqrt(v[0]**2 + v[1]**2) for v in velocities]
        
        # Find when robot stops moving (velocity near zero)
        velocity_threshold = 1e-10
        last_moving_idx = None
        for i in range(len(resultant_velocities) - 1, 0, -1):
            if resultant_velocities[i] > velocity_threshold:
                last_moving_idx = i
                break
        
        if last_moving_idx is not None:
            completion_time = last_moving_idx * time_step
            completion_times.append(completion_time)
            print(f"Agent {agent_id} completion time: {completion_time:.2f}s at step {last_moving_idx}")
        else:
            # If no movement found, use the last timestep
            completion_time = (len(velocities) - 1) * time_step
            completion_times.append(completion_time)
            print(f"Agent {agent_id} never moved significantly, using final timestep")
    
    # Calculate makespan ratio (only for dynamic agents)
    if len(completion_times) >= 2:
        fastest_time = min(completion_times)
        slowest_time = max(completion_times)
        if fastest_time > 0:
            makespan_ratio = slowest_time / fastest_time
            print("\n" + "="*65)
            print("MAKESPAN RATIO")
            print("="*65)
            print(f"Fastest Agent Time: {fastest_time:.4f} seconds")
            print(f"Slowest Agent Time: {slowest_time:.4f} seconds")
            print(f"Makespan Ratio: {makespan_ratio:.4f}")
            print("="*65 + "\n")
    
    # Evaluate each moving agent (agent_data already contains only moving agents)
    for agent in agent_data:
        agent_id = agent['id']
        positions = agent['positions']
        velocities = agent['velocities']
        start_pos = agent['start_pos']
        goal_pos = agent['goal_pos']
        
        print(f"\nRobot {agent_id} Metrics:")
        print("-" * 40)
        
        # Path deviation metrics
        if positions:
            avg_dev, max_dev, actual_length, nominal_length = calculate_path_deviation(
                positions, start_pos, goal_pos)
            
            print(f"  Average Path Deviation: {avg_dev:.4f} units")
            print(f"  Maximum Path Deviation: {max_dev:.4f} units")
            print(f"  Actual Path Length: {actual_length:.4f} units")
            print(f"  Nominal Path Length: {nominal_length:.4f} units")
            print(f"  Path Efficiency: {(nominal_length/actual_length)*100:.2f}%" if actual_length > 0 else "  Path Efficiency: N/A")
            
            total_avg_deviation += avg_dev
            total_max_deviation = max(total_max_deviation, max_dev)
        else:
            print("  No position data available")
        
        # Velocity metrics
        if velocities:
            avg_delta_vel = calculate_average_delta_velocity(velocities)
            print(f"  Average Delta Velocity: {avg_delta_vel:.4f} units/s")
            total_avg_delta_velocity += avg_delta_vel
        else:
            print("  No velocity data available")
    
    # Overall metrics for moving robots
    num_moving_agents = len(agent_data)
    if num_moving_agents > 0:
        print(f"\nOVERALL METRICS:")
        print("-" * 40)
        print(f"  Average Path Deviation: {total_avg_deviation/num_moving_agents:.4f} units")
        print(f"  Maximum Path Deviation: {total_max_deviation:.4f} units")
        print(f"  Average Delta Velocity: {total_avg_delta_velocity/num_moving_agents:.4f} units/s")
        
        # Calculate makespan and flow rate
        gap_width = get_gap_width(scenario_type)
        makespan = max(completion_times) if completion_times else 0.0
        flow_rate = calculate_flow_rate(agent_data, time_step, gap_width)
        
        print(f"  Makespan: {makespan:.2f} seconds")
        print(f"  Gap Width: {gap_width:.1f} meters")
        print(f"  Flow Rate: {flow_rate:.6f} agents/(second·meter)")
        
        # Success rate
        successful_agents = 0
        for agent in agent_data:
            if agent['positions']:
                final_pos = agent['positions'][-1]
                goal_pos = agent['goal_pos']
                final_distance = np.linalg.norm(np.array(final_pos) - np.array(goal_pos))
                if final_distance <= 0.5:  # Goal tolerance
                    successful_agents += 1
        
        success_rate = (successful_agents / num_moving_agents) * 100
        print(f"  Success Rate: {success_rate:.1f}% ({successful_agents}/{num_moving_agents} robots)")
    
    print("="*80)
    return {
        'avg_path_deviation': total_avg_deviation/num_moving_agents if num_moving_agents > 0 else 0,
        'max_path_deviation': total_max_deviation,
        'avg_delta_velocity': total_avg_delta_velocity/num_moving_agents if num_moving_agents > 0 else 0,
        'makespan': slowest_time if len(completion_times) >= 2 else 0,
        'gap_width': gap_width,
        'flow_rate': flow_rate if num_moving_agents > 0 else 0,
        'success_rate': success_rate if num_moving_agents > 0 else 0,
        'num_moving_agents': num_moving_agents,
        'makespan_ratio': makespan_ratio if len(completion_times) >= 2 else 1.0
    }

def main():
    args = parse_cli_args()
    
    if not args.quiet:
        print("=== Social Collision Avoidance Scenario Runner ===")
    
    # Determine if we're in CLI mode or interactive mode
    cli_mode = args.scenario is not None
    
    if cli_mode:
        # CLI mode - use command line arguments
        scenario_type = args.scenario
        
        # Get agents from various sources
        user_agents = []
        
        if args.agents_file:
            # Load from file
            user_agents = load_agents_from_file(args.agents_file)
            if not args.quiet:
                print(f"Loaded {len(user_agents)} agents from {args.agents_file}")
        
        elif args.agents is not None:
            # Check for individual agent arguments
            agent_found = False
            for i in range(1, min(args.agents + 1, 11)):
                agent_arg = getattr(args, f'agent{i}', None)
                if agent_arg:
                    try:
                        agent = parse_agent_string(agent_arg, scenario_type, i)
                        user_agents.append(agent)
                        agent_found = True
                    except ValueError as e:
                        print(f"Error parsing agent {i}: {e}")
                        return 1
            
            # If no individual agents specified, create defaults
            if not agent_found:
                user_agents = create_default_agents(scenario_type, args.agents)
                if not args.quiet:
                    print(f"Created {len(user_agents)} default agents for {scenario_type} scenario")
        
        else:
            # No agents specified
            if not args.quiet:
                print("Running default scenario with no additional agents")
        
        if not args.quiet and user_agents:
            print(f"\nAgent configuration:")
            for i, (sx, sy, gx, gy, r, s, h) in enumerate(user_agents):
                print(f"  Agent {i+1}: Start=({sx:.1f}, {sy:.1f}), Goal=({gx:.1f}, {gy:.1f}), "
                      f"R={r:.1f}, Speed={s:.1f}, Heading={h:.2f}")
    
    else:
        # Interactive mode - use prompts
        scenario_type = get_scenario_choice()
        num_agents = get_num_agents()
        
        if num_agents > 0:
            user_agents = get_agent_parameters(num_agents, scenario_type)
        else:
            user_agents = []
            print("Running default scenario with no additional agents")
    
    # Run the scenario
    try:
        evaluation_results = run_scenario(scenario_type, user_agents, 
                                 num_steps=getattr(args, 'steps', 100))
        
        if not args.quiet:
            print("Scenario completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"Error running scenario: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 