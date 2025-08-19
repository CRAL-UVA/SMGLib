import numpy as np
import sys
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
from test import PLAN
from plot import plot_trajectory
from pathlib import Path

# Import standardized environment configuration
sys.path.append(str(Path(__file__).resolve().parents[3] / 'src'))
from utils import StandardizedEnvironment

def get_input(prompt, default, type_cast=str):
    while True:
        user_input = input(f"{prompt} (default: {default}): ")
        if not user_input:
            return default
        try:
            return type_cast(user_input)
        except ValueError:
            print(f"Invalid input! Please enter a valid {type_cast.__name__}.")

def _impc_logs_dirs():
    root_dir = Path(__file__).resolve().parents[3]
    anim_dir = root_dir / 'logs' / 'Social-IMPC-DR' / 'animations'
    traj_dir = root_dir / 'logs' / 'Social-IMPC-DR' / 'trajectories'
    anim_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    return anim_dir, traj_dir


def save_video(frames, filename=None, fps=5, scenario_type='impc', agent_summary='default'):
    # This function is kept for compatibility but no longer saves video files
    # Only GIF files are generated now
    pass


def save_gif(agent_list, r_min, filename=None, fps=5, num_moving_agents=None, scenario_type='impc', agent_summary='default'):
    """Save animation as GIF file."""
    anim_dir, _ = _impc_logs_dirs()
    if filename is None:
        filename = anim_dir / f"{scenario_type}_{agent_summary}agents.gif"
    else:
        filename = anim_dir / filename

    fig, ax = plt.subplots(figsize=StandardizedEnvironment.FIG_SIZE)
    ax.set_xlim(StandardizedEnvironment.GRID_X_MIN, StandardizedEnvironment.GRID_X_MAX)
    ax.set_ylim(StandardizedEnvironment.GRID_Y_MIN, StandardizedEnvironment.GRID_Y_MAX)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Build positions per frame from agent_list
    max_frames = max(len(a.position) for a in agent_list) if agent_list else 0

    # Use num_moving_agents parameter to properly distinguish between dynamic agents and obstacles
    if num_moving_agents is None:
        # Fallback: infer dynamic/obstacles if num_moving_agents not provided
        def is_stationary(a):
            return len(a.position) > 1 and all(np.allclose(a.position[0], p) for p in a.position)
        dynamic_indices = [i for i, a in enumerate(agent_list) if not is_stationary(a)]
        obstacle_indices = [i for i in range(len(agent_list)) if i not in dynamic_indices]
    else:
        # Use the provided num_moving_agents parameter
        dynamic_indices = list(range(num_moving_agents))
        obstacle_indices = list(range(num_moving_agents, len(agent_list)))

    # Use standardized colors
    colors = StandardizedEnvironment.AGENT_COLORS

    dyn_scatter = ax.scatter([], [], c=[], s=200, edgecolors='black', linewidths=1, label='Agent')
    obs_scatter = ax.scatter([], [], c='gray', s=200, edgecolors='black', linewidths=1, label='Obstacle')

    # Goals as green stars - only for dynamic agents
    goal_points = []
    for i, a in enumerate(agent_list):
        if i < num_moving_agents:  # Only dynamic agents have goals
            # IMPC-DR agents use 'target' attribute, not 'goal'
            gp = getattr(a, 'target', None)
            if gp is not None and len(gp) == 2:
                goal_points.append(gp)
            else:
                goal_points.append(None)
        else:
            goal_points.append(None)
    
    # Plot goals as green stars
    for gp in goal_points:
        if gp is not None:
            ax.scatter(gp[0], gp[1], marker='*', s=300, color='green', edgecolor='black', zorder=4)

    # Legend matching CADRL
    legend_handles = []
    legend_labels = []
    legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                        markersize=10, markerfacecolor='gray', markeredgecolor='black'))
    legend_labels.append('Obstacle')
    for i, _ in enumerate(dynamic_indices):
        color = colors[i % len(colors)]
        legend_handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                            markersize=10, markerfacecolor=color, markeredgecolor='black'))
        legend_labels.append(f'Agent {i+1}')
    legend_handles.append(mlines.Line2D([], [], color='green', marker='*', linestyle='None',
                                        markersize=12, markerfacecolor='green', markeredgecolor='none'))
    legend_labels.append('Goal')

    ax.legend(legend_handles, legend_labels,
              loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12, borderaxespad=0., markerscale=1.2)
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

    def frame_positions(frame):
        pos = []
        for a in agent_list:
            if frame < len(a.position):
                pos.append(a.position[frame])
            else:
                pos.append(a.position[-1])
        return pos

    def animate(frame):
        pos = frame_positions(frame)
        # Obstacles (all agents after num_moving_agents)
        obs_pos = [pos[i] for i in obstacle_indices] if obstacle_indices else []
        if obs_pos:
            obs_scatter.set_offsets(np.array(obs_pos).reshape(-1, 2))
        else:
            obs_scatter.set_offsets(np.empty((0, 2)))
        # Dynamic agents (first num_moving_agents)
        dyn_pos = [pos[i] for i in dynamic_indices] if dynamic_indices else []
        if dyn_pos:
            dyn_colors = [colors[i % len(colors)] for i in range(len(dyn_pos))]
            dyn_scatter.set_offsets(np.array(dyn_pos).reshape(-1, 2))
            dyn_scatter.set_color(dyn_colors)
        else:
            dyn_scatter.set_offsets(np.empty((0, 2)))
            dyn_scatter.set_color([])
        return [dyn_scatter, obs_scatter]

    anim = FuncAnimation(fig, animate, frames=max_frames, interval=StandardizedEnvironment.ANIMATION_INTERVAL, blit=True)
    anim.save(str(filename), writer='pillow', fps=StandardizedEnvironment.ANIMATION_FPS)
    print(f"GIF animation saved as {filename}")

    plt.close(fig)


def generate_animation(agent_list, r_min, filename=None, num_moving_agents=None, scenario_type='impc', agent_summary=None):
    # Only generate GIF animation, no video or frame files
    if agent_summary is None:
        agent_summary = f"{len(agent_list)}"

    # Save only as GIF
    save_gif(agent_list, r_min, filename=None, fps=StandardizedEnvironment.ANIMATION_FPS, num_moving_agents=num_moving_agents,
             scenario_type=scenario_type, agent_summary=agent_summary)

def setup_doorway_scenario():
    """Sets up the stationary wall for the doorway scenario using standardized configuration."""
    print("Setting up Doorway Environment using standardized configuration...")
    
    # Use standardized doorway obstacles
    obstacles = StandardizedEnvironment.get_doorway_obstacles()
    
    ini_x_obstacles = obstacles
    ini_v_obstacles = [np.zeros(2) for _ in ini_x_obstacles]
    # Stationary agents have their target set to their start position
    target_obstacles = ini_x_obstacles 

    return ini_x_obstacles, ini_v_obstacles, target_obstacles

def setup_hallway_scenario():
    """Sets up the stationary walls for the hallway scenario using standardized configuration."""
    print("Setting up Hallway Environment using standardized configuration...")
    
    # Use standardized hallway obstacles
    obstacles = StandardizedEnvironment.get_hallway_obstacles()
    
    ini_x_obstacles = obstacles
    ini_v_obstacles = [np.zeros(2) for _ in ini_x_obstacles]
    # Stationary agents have their target set to their start position
    target_obstacles = ini_x_obstacles
    
    return ini_x_obstacles, ini_v_obstacles, target_obstacles

def setup_intersection_scenario():
    """Sets up the stationary walls for the intersection scenario using standardized configuration."""
    print("Setting up Intersection Environment using standardized configuration...")
    
    # Use standardized intersection obstacles
    obstacles = StandardizedEnvironment.get_intersection_obstacles()
    
    ini_x_obstacles = obstacles
    ini_v_obstacles = [np.zeros(2) for _ in ini_x_obstacles]
    # Stationary agents have their target set to their start position
    target_obstacles = ini_x_obstacles
    
    return ini_x_obstacles, ini_v_obstacles, target_obstacles

def main():
    env_type = None
    verbose_mode = True  # Default to verbose for backwards compatibility
    
    if len(sys.argv) > 1:
        env_type = sys.argv[1]
    
    if len(sys.argv) > 2:
        verbose_arg = sys.argv[2]
        verbose_mode = (verbose_arg == '--verbose')

    obstacle_agents_x = []
    obstacle_agents_v = []
    obstacle_agents_target = []
    
    if env_type == 'doorway':
        obstacle_agents_x, obstacle_agents_v, obstacle_agents_target = setup_doorway_scenario()
    elif env_type == 'hallway':
        obstacle_agents_x, obstacle_agents_v, obstacle_agents_target = setup_hallway_scenario()
    elif env_type == 'intersection':
        obstacle_agents_x, obstacle_agents_v, obstacle_agents_target = setup_intersection_scenario()
    
    # --- Get User Input for Simulation ---

    # Get parameters for the moving drones
    num_moving_drones = get_input("Enter number of moving drones", 2, int)
    
    # Get simulation parameters from user - optimized per environment
    min_radius = get_input("Enter minimum distance between drones", StandardizedEnvironment.DEFAULT_COLLISION_DISTANCE, float)
    
    # Environment-specific parameter optimization
    if env_type == 'doorway':
        # Doorway: Conservative parameters for precision
        wall_collision_multiplier = get_input("Enter wall collision distance multiplier (1.2-2.0 recommended)", 1.5, float)
        epsilon = get_input("Enter epsilon value", 0.05, float)
        step_size = get_input("Enter step size", 0.05, float)
        k_value = get_input("Enter k value", 15, int)
        max_steps = get_input("Enter maximum number of steps", 150, int)
    elif env_type == 'hallway':
        # Hallway: Balanced parameters for performance
        wall_collision_multiplier = get_input("Enter wall collision distance multiplier (1.0-1.5 recommended)", 1.2, float)
        epsilon = get_input("Enter epsilon value", 0.1, float)  # Increased for faster convergence
        step_size = get_input("Enter step size", 0.1, float)  # Increased for faster movement
        k_value = get_input("Enter k value", 10, int)  # Reduced for faster computation
        max_steps = get_input("Enter maximum number of steps", 100, int)  # Reduced for faster completion
    elif env_type == 'intersection':
        # Intersection: Aggressive parameters for performance
        wall_collision_multiplier = get_input("Enter wall collision distance multiplier (1.0-1.3 recommended)", 1.1, float)
        epsilon = get_input("Enter epsilon value", 0.15, float)  # Increased for faster convergence
        step_size = get_input("Enter step size", 0.15, float)  # Increased for faster movement
        k_value = get_input("Enter k value", 8, int)  # Reduced for faster computation
        max_steps = get_input("Enter maximum number of steps", 80, int)  # Reduced for faster completion
    
    print("\nConfigure moving drones:")
    
    # Print environment-specific instructions using standardized coordinates
    if env_type == 'doorway':
        print("\nDoorway Configuration:")
        print("- The doorway has a vertical wall at x=0 with a gap between y=-2 and y=2")
        print("- X coordinates should be between -5 and 5")
        print("- Y coordinates should be between -7 and 7")
    elif env_type == 'hallway':
        print("\nHallway Configuration:")
        print("- The hallway has walls at y=-2 and y=2")
        print("- Robots should stay between y=-1.5 and y=1.5 (middle of hallway)")
        print("- X coordinates should be between -5 and 5")
    elif env_type == 'intersection':
        print("\nIntersection Configuration:")
        print("- The intersection has corridors with center at (0, 0)")
        print("- Corridor width extends from -2 to 2 in both directions")
        print("- X and Y coordinates should be between -5 and 5")
    
    # Get drone positions in ORCA-style individual configuration
    ini_x_moving = []
    target_moving = []
    
    # Get standardized default positions
    standard_positions = StandardizedEnvironment.get_standard_agent_positions(env_type, num_moving_drones)
    
    # Convert to the format expected by the rest of the code
    default_positions = []
    for pos in standard_positions:
        default_positions.append({
            'start_x': pos['start'][0],
            'start_y': pos['start'][1],
            'goal_x': pos['goal'][0],
            'goal_y': pos['goal'][1]
        })
    
    for i in range(num_moving_drones):
        print(f"\n--- Agent {i+1} Parameters ---")
        
        # Get default values for this drone (cycle through available defaults)
        default_idx = i % len(default_positions)
        defaults = default_positions[default_idx]
        
        # Get start position
        start_x = get_input(f"Start X position (default: {defaults['start_x']})", defaults['start_x'], float)
        start_y = get_input(f"Start Y position (default: {defaults['start_y']})", defaults['start_y'], float)
        
        # Get goal position  
        goal_x = get_input(f"Goal X position (default: {defaults['goal_x']})", defaults['goal_x'], float)
        goal_y = get_input(f"Goal Y position (default: {defaults['goal_y']})", defaults['goal_y'], float)
        
        # Store positions
        ini_x_moving.append(np.array([start_x, start_y]))
        target_moving.append(np.array([goal_x, goal_y]))
        
        print(f"Agent {i+1} configured: Start=({start_x}, {start_y}), Goal=({goal_x}, {goal_y})")
    
    ini_v_moving = [np.zeros(2) for _ in range(num_moving_drones)]

    # --- Combine moving and stationary agents ---
    ini_x = ini_x_moving + obstacle_agents_x
    ini_v = ini_v_moving + obstacle_agents_v
    target = target_moving + obstacle_agents_target
    num_drones = len(ini_x)
    
    print("\nStarting simulation...")
    result, agent_list, completion_step = PLAN(num_drones, ini_x, ini_v, target, min_radius, epsilon, step_size, k_value, max_steps, num_moving_drones=num_moving_drones, wall_collision_multiplier=wall_collision_multiplier, verbose=verbose_mode)
    
    # Save completion step for Flow Rate calculation
    with open("completion_step.txt", "w") as f:
        f.write(str(completion_step))
    
    if result:
        print("\nSimulation completed successfully!")
        print("Using standardized environment configuration for consistent visualization.")
        generate_animation(agent_list, min_radius, num_moving_agents=num_moving_drones, scenario_type=env_type, agent_summary=str(num_moving_drones))
    else:
        print("\nSimulation failed to find a solution.")
    
    # Generate and save trajectory animation
    plot_trajectory(agent_list, min_radius)
    print("Trajectory animation saved as 'trajectory.svg'")
    
    # Output results
    print("\nSimulation Results:")
    print(f"Number of steps taken: {len(agent_list[0].position)}")
    print(f"Final positions:")
    for i, agent in enumerate(agent_list):
        if i < num_moving_drones:
            print(f"Moving Drone {i+1}: {agent.position[-1]}")
    print(f"Final velocities:")
    for i, agent in enumerate(agent_list):
        if i < num_moving_drones:
            print(f"Moving Drone {i+1}: {agent.v[-1]}")

if __name__ == "__main__":
    main()