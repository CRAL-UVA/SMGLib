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
from envs.policies.StaticPolicy import StaticPolicy

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
    """Create the base agents for the intersection scenario (static obstacles)"""
    agents = []
    # Add static agents around the intersection - 24 obstacles total
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
            StaticPolicy,  # Use StaticPolicy for static obstacles
            UnicycleDynamics,
            [OtherAgentsStatesSensor],
            i
        ))
    return agents

def create_base_hallway_agents():
    """Create the base agents for the hallway scenario (static obstacles)"""
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
            StaticPolicy,  # Use StaticPolicy for static obstacles
            UnicycleDynamics,
            [OtherAgentsStatesSensor],
            i
        ))
    return agents

def create_base_doorway_agents():
    """Create the base agents for the doorway scenario (horizontal obstacles)"""
    agents = []
    # Add static agents forming horizontal wall with doorway gap
    # Single horizontal line at y=0.0 with gap in the middle
    positions = []
    
    # Left side of horizontal wall: x from -9.0 to -2.0
    for x in [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0]:
        positions.append((x, 0.0))
    
    # Right side of horizontal wall: x from 2.0 to 9.0  
    for x in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
        positions.append((x, 0.0))
    
    for i, (x, y) in enumerate(positions):
        agents.append(Agent(
            x, y, x, y,  # Static agents have same start and goal
            0.5, 1.0, np.pi,
            StaticPolicy,  # Use StaticPolicy for static obstacles
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
        # Create horizontal wall aligned across y=0 with same spacing
        for x in [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0]:
            obstacles.append((x, 0.0))
        for x in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
            obstacles.append((x, 0.0))
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
        # First agent: bottom side going through doorway
        defaults['start_x'] = 0.0
        defaults['start_y'] = -5.0
        defaults['goal_x'] = 0.0
        defaults['goal_y'] = 9.0
        defaults['radius'] = 0.5
        defaults['pref_speed'] = 1.0
        defaults['heading'] = np.pi/2
        
        # Second agent: top side going through doorway
        defaults['agent2'] = {
            'start_x': 0.0,
            'start_y': 9.0,
            'goal_x': 0.0,
            'goal_y': -5.0,
            'radius': 0.5,
            'pref_speed': 1.0,
            'heading': -np.pi/2
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
            # Check if this is actually a static agent (same start and goal)
            is_static = np.allclose([start_x, start_y], [goal_x, goal_y], atol=1e-6)
            
            if is_static:
                print(f"Warning: Agent {len(agents)} has same start and goal positions, using StaticPolicy")
                policy = StaticPolicy
            else:
                policy = tc.policy_dict["CADRL"]
            
            print(f"Agent {i+1}: Start=({start_x:.1f}, {start_y:.1f}), Goal=({goal_x:.1f}, {goal_y:.1f})")
            
            new_agent = Agent(
                start_x, start_y, goal_x, goal_y,
                radius, pref_speed, heading,
                policy,
                UnicycleDynamics,
                [OtherAgentsStatesSensor],
                len(agents)
            )
            
            agents.append(new_agent)
            if not is_static:
                dynamic_agents.append(new_agent)

    # Set config for the number of agents
    Config.MAX_NUM_AGENTS_IN_ENVIRONMENT = len(agents)
    
    # Categorize agents by their behavior
    all_dynamic_agents = []
    all_static_agents = []
    for agent in agents:
        # Check if agent has different start and goal positions
        is_dynamic = not np.allclose(agent.pos_global_frame, agent.goal_global_frame, atol=1e-6)
        if is_dynamic:
            all_dynamic_agents.append(agent)
        else:
            all_static_agents.append(agent)
    
    print(f"Total agents: {len(agents)}")
    print(f"Dynamic agents: {len(all_dynamic_agents)} (agents that should move)")
    print(f"Static agents: {len(all_static_agents)} (obstacles/agents that stay in place)")
    
    # Validate that dynamic agents have proper policies for movement
    for i, agent in enumerate(all_dynamic_agents):
        policy_name = getattr(agent.policy, 'str', str(type(agent.policy).__name__))
        if policy_name == 'StaticPolicy':
            print(f"⚠️  WARNING: Dynamic agent {agent.id} (different start/goal) has StaticPolicy - this agent won't move!")
        else:
            start_pos = agent.pos_global_frame
            goal_pos = agent.goal_global_frame
            distance = np.linalg.norm(goal_pos - start_pos)
            print(f"✓ Dynamic agent {agent.id}: Start=({start_pos[0]:.1f}, {start_pos[1]:.1f}), "
                  f"Goal=({goal_pos[0]:.1f}, {goal_pos[1]:.1f}), Distance={distance:.2f}, Policy={policy_name}")
    
    # Validate that static agents have proper policies
    for i, agent in enumerate(all_static_agents):
        policy_name = getattr(agent.policy, 'str', str(type(agent.policy).__name__))
        if policy_name != 'StaticPolicy':
            print(f"⚠️  INFO: Static agent {agent.id} (same start/goal) has {policy_name} policy - "
                  f"recommend using StaticPolicy for better performance")
    
    print(f"Config MAX_NUM_AGENTS: {Config.MAX_NUM_AGENTS_IN_ENVIRONMENT}")

    # Create the environment
    env = gym.make("CollisionAvoidance-v0")

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
    
    # Store positions for animation
    positions_history = []
    
    print(f"\n🚀 Starting simulation with {len(all_dynamic_agents)} moving agents for {num_steps} steps...")
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
        
        # Check if any dynamic agents have reached their goals
        agents_at_goal = []
        agents_still_moving = []
        for j, agent in enumerate(dynamic_agents):
            distance_to_goal = np.linalg.norm(agent.goal_global_frame - agent.pos_global_frame)
            if distance_to_goal <= 0.3:  # More lenient goal tolerance
                agents_at_goal.append(agent)
            else:
                agents_still_moving.append(agent)
            
            # Optional debug output every 10 steps (disabled for cleaner output)
            # if i % 10 == 0 and i > 0:
            #     print(f"Step {i}, Agent {j+1}: Pos=({agent.pos_global_frame[0]:.2f}, {agent.pos_global_frame[1]:.2f}), "
            #           f"Goal=({agent.goal_global_frame[0]:.2f}, {agent.goal_global_frame[1]:.2f}), "
            #           f"Distance={distance_to_goal:.2f}")
        
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

    print(f"\n✅ Simulation completed after {i+1} steps")
    
    # Check final status of dynamic agents
    agents_at_goal = 0
    for agent in all_dynamic_agents:
        distance_to_goal = np.linalg.norm(agent.goal_global_frame - agent.pos_global_frame)
        if distance_to_goal < 0.5:  # Within 0.5 units of goal
            agents_at_goal += 1
            print(f"✓ Agent {agent.id} reached goal (distance: {distance_to_goal:.2f})")
        else:
            print(f"✗ Agent {agent.id} did not reach goal (distance: {distance_to_goal:.2f})")
    
    print(f"\nFinal Result: {agents_at_goal}/{len(all_dynamic_agents)} dynamic agents reached their goals")

    # Create animation
    print("Creating animation...")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True)

    # Define colors for multiple agents
    agent_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Create scatter plot for dynamic agents with different colors (filled with outlines)
    agent_scatter = ax.scatter([], [], c=[], s=200, edgecolors='black', linewidths=1, label='Agents')
    
    # Plot static obstacles as solid black circles
    if obstacle_coords:
        obstacle_x = [coord[0] for coord in obstacle_coords]
        obstacle_y = [coord[1] for coord in obstacle_coords]
        ax.scatter(obstacle_x, obstacle_y, c='black', s=200, edgecolors='none', alpha=0.8, label='Obstacle')

    # Create goal markers for dynamic agents with matching colors (for all scenarios)
    for i, agent in enumerate(dynamic_agents):
        goal = agent.goal_global_frame
        color = agent_colors[i % len(agent_colors)]
        ax.plot(goal[0], goal[1], '*', color=color, markersize=15, 
                label=f'Goal {i+1}' if len(dynamic_agents) > 1 else 'Goal')

    # Custom legend handles for all scenarios
    legend_handles = []
    legend_labels = []
    
    # Add agent handles with different colors if multiple agents
    if len(dynamic_agents) > 1:
        for i, agent in enumerate(dynamic_agents):
            color = agent_colors[i % len(agent_colors)]
            agent_handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', 
                                       markersize=10, markerfacecolor=color, markeredgecolor='black')
            legend_handles.append(agent_handle)
            legend_labels.append(f'Agent {i+1}')
    else:
        agent_handle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', 
                                   markersize=10, markerfacecolor='blue', markeredgecolor='black')
        legend_handles.append(agent_handle)
        legend_labels.append('Agent')
    
    # Add obstacle handle
    obstacle_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', 
                                  markersize=10, markerfacecolor='black', markeredgecolor='none')
    legend_handles.append(obstacle_handle)
    legend_labels.append('Obstacle')
    
    # Add goal handles with matching colors
    if len(dynamic_agents) > 1:
        for i, agent in enumerate(dynamic_agents):
            color = agent_colors[i % len(agent_colors)]
            goal_handle = mlines.Line2D([], [], color=color, marker='*', linestyle='None', 
                                      markersize=12, markerfacecolor=color, markeredgecolor='none')
            legend_handles.append(goal_handle)
            legend_labels.append(f'Goal {i+1}')
    else:
        goal_handle = mlines.Line2D([], [], color='blue', marker='*', linestyle='None', 
                                  markersize=12, markerfacecolor='blue', markeredgecolor='none')
        legend_handles.append(goal_handle)
        legend_labels.append('Goal')

    # Place legend outside the plot
    ax.legend(
        legend_handles, legend_labels,
        loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12, borderaxespad=0., markerscale=1.2
    )

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

    # Animation update function - only animate dynamic agents with colors
    def update(frame):
        if frame < len(positions_history):
            # Get positions of dynamic agents only
            dynamic_agent_positions = []
            colors = []
            for i, agent in enumerate(dynamic_agents):
                # Find this dynamic agent's index in the full agents list
                agent_index = agents.index(agent)
                pos = positions_history[frame][agent_index]
                dynamic_agent_positions.append(pos)
                colors.append(agent_colors[i % len(agent_colors)])
            
            # Update agent scatter plot with dynamic agents only
            if dynamic_agent_positions:
                agent_scatter.set_offsets(np.array(dynamic_agent_positions).reshape(-1, 2))
                agent_scatter.set_color(colors)
            else:
                agent_scatter.set_offsets(np.empty((0, 2)))
                agent_scatter.set_color([])
                
        return [agent_scatter]

    # Create animation with slower speed for better visibility
    anim = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=150, blit=True)

    # Save animation in the correct animations directory with unique name
    # Save to root logs/Social-CADRL/animations
    from pathlib import Path
    root_dir = Path(__file__).resolve().parents[5]
    animations_dir = root_dir / 'logs' / 'Social-CADRL' / 'animations'
    os.makedirs(animations_dir, exist_ok=True)
    
    # Create unique filename based on configuration
    agent_count = len([a for a in agents if hasattr(a, 'policy')])
    agent_summary = f"{agent_count}agents"
    if user_agents and len(user_agents) > 0:
        # Add position summary for first 2 agents
        if len(user_agents) >= 1:
            sx, sy, gx, gy = user_agents[0][:4]
            agent_summary += f"_s{sx:.0f}_{sy:.0f}_g{gx:.0f}_{gy:.0f}"
        if len(user_agents) >= 2:
            sx, sy, gx, gy = user_agents[1][:4]
            agent_summary += f"_s{sx:.0f}_{sy:.0f}_g{gx:.0f}_{gy:.0f}"
    else:
        agent_summary += "_default"
    
    filename = str(animations_dir / f"{scenario_type}_{agent_summary}.gif")
    
    # Save animation
    anim.save(filename, writer='pillow')
    print(f"Animation saved as {filename}")

    # Don't show plot to prevent freezing
    plt.close()

    return env, agents

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
        env, agents = run_scenario(scenario_type, user_agents, 
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