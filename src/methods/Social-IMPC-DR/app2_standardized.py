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
    anim_dir, _ = _impc_logs_dirs()
    if filename is None:
        filename = anim_dir / f"{scenario_type}_{agent_summary}agents.avi"
    else:
        filename = anim_dir / filename
    if not frames:
        print("No frames captured. Cannot save video.")
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved as {filename}")

def save_gif_standardized(agent_list, r_min, filename=None, fps=5, num_moving_agents=None, scenario_type='impc', agent_summary='default'):
    """Save animation as GIF file using standardized environment configuration."""
    anim_dir, _ = _impc_logs_dirs()
    if filename is None:
        filename = anim_dir / f"{scenario_type}_{agent_summary}agents.gif"
    else:
        filename = anim_dir / filename

    # Use standardized plot creation
    fig, ax = StandardizedEnvironment.create_standard_plot(scenario_type, show_obstacles=True)
    
    # Build positions per frame from agent_list
    max_frames = max(len(a.position) for a in agent_list) if agent_list else 0

    # Infer dynamic/obstacles if available: treat stationary agents as obstacles
    def is_stationary(a):
        return len(a.position) > 1 and all(np.allclose(a.position[0], p) for p in a.position)

    dynamic_indices = [i for i, a in enumerate(agent_list) if not is_stationary(a)]
    obstacle_indices = [i for i in range(len(agent_list)) if i not in dynamic_indices]

    # Use standardized colors
    colors = StandardizedEnvironment.AGENT_COLORS

    dyn_scatter = ax.scatter([], [], c=[], s=200, edgecolors='black', linewidths=1, label='Agent')
    obs_scatter = ax.scatter([], [], c='gray', s=200, edgecolors='black', linewidths=1, label='Obstacle')

    # Goals as green stars if each agent has a goal attribute 'goal'
    goal_points = []
    for a in agent_list:
        gp = getattr(a, 'goal', None)
        if gp is not None and len(gp) == 2:
            goal_points.append(gp)
        else:
            goal_points.append(None)
    for gp in goal_points:
        if gp is not None:
            ax.plot(gp[0], gp[1], '*', color='green', markersize=12)

    # Create standardized legend
    legend_handles, legend_labels = StandardizedEnvironment.create_standard_legend(len(dynamic_indices))
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
        # Obstacles
        obs_pos = [pos[i] for i in obstacle_indices] if obstacle_indices else []
        if obs_pos:
            obs_scatter.set_offsets(np.array(obs_pos).reshape(-1, 2))
        else:
            obs_scatter.set_offsets(np.empty((0, 2)))
        # Dynamic agents
        dyn_pos = [pos[i] for i in dynamic_indices] if dynamic_indices else []
        if dyn_pos:
            dyn_colors = [colors[i % len(colors)] for i in range(len(dyn_pos))]
            dyn_scatter.set_offsets(np.array(dyn_pos).reshape(-1, 2))
            dyn_scatter.set_color(dyn_colors)
        else:
            dyn_scatter.set_offsets(np.empty((0, 2)))
            dyn_scatter.set_color([])
        return [dyn_scatter, obs_scatter]

    anim = FuncAnimation(fig, animate, frames=max_frames, 
                        interval=StandardizedEnvironment.ANIMATION_INTERVAL, blit=True)
    anim.save(str(filename), writer='pillow', fps=fps)
    print(f"GIF animation saved as {filename}")

    try:
        html_name = str(filename).replace('.gif', '.html')
        anim.save(html_name, writer='html')
        print(f"HTML animation saved as {html_name}")
    except Exception:
        pass

    plt.close(fig)

def generate_animation_standardized(agent_list, r_min, filename=None, num_moving_agents=None, scenario_type='impc', agent_summary=None):
    """Generate animation using standardized environment configuration."""
    frames = []
    
    # Use standardized plot creation
    fig, ax = StandardizedEnvironment.create_standard_plot(scenario_type, show_obstacles=True)
    
    # Use standardized colors
    colors = StandardizedEnvironment.AGENT_COLORS
    
    for step in range(len(agent_list[0].position)):
        ax.clear()
        
        # Reset plot with standardized settings
        ax.set_xlim(StandardizedEnvironment.GRID_X_MIN, StandardizedEnvironment.GRID_X_MAX)
        ax.set_ylim(StandardizedEnvironment.GRID_Y_MIN, StandardizedEnvironment.GRID_Y_MAX)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        
        # Add obstacles
        if scenario_type == 'doorway':
            obstacles = StandardizedEnvironment.get_doorway_obstacles()
        elif scenario_type == 'hallway':
            obstacles = StandardizedEnvironment.get_hallway_obstacles()
        elif scenario_type == 'intersection':
            obstacles = StandardizedEnvironment.get_intersection_obstacles()
        else:
            obstacles = []
        
        for obs in obstacles:
            circle = Circle(obs, radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, 
                          facecolor='gray', edgecolor='black', alpha=0.7)
            ax.add_patch(circle)
        
        for i, agent in enumerate(agent_list):
            if num_moving_agents is not None:
                if i < num_moving_agents:
                    color = colors[i % len(colors)]
                else:
                    color = 'gray'
            else:
                color = colors[i % len(colors)]
            
            # Use last known position if step is out of bounds
            pos_index = min(step, len(agent.position) - 1)
            pos = agent.position[pos_index]
            
            # Draw trajectory as a dotted line
            if pos_index > 0:
                past_positions = np.array(agent.position[:pos_index+1])
                ax.plot(past_positions[:, 0], past_positions[:, 1], linestyle="dotted", color=color, linewidth=2)
            
            # Draw solid line for completed part of the trajectory
            if pos_index > 5:
                completed_positions = np.array(agent.position[max(0, pos_index-5):pos_index+1])
                ax.plot(completed_positions[:, 0], completed_positions[:, 1], linestyle="solid", color=color, linewidth=2)

            # Draw drone as a circle using standardized radius
            circle = Circle(pos, radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, 
                          edgecolor='black', facecolor=color, zorder=3)
            ax.add_patch(circle)

            # Mark start position with a square
            ax.scatter(agent.position[0][0], agent.position[0][1], marker='s', s=200, 
                      edgecolor='black', facecolor=color)

            # Mark target with a diamond
            if num_moving_agents is None or i < num_moving_agents:
                ax.scatter(agent.target[0], agent.target[1], marker='*', s=300, 
                          edgecolor='black', facecolor=color)
        
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(image)

    if agent_summary is None:
        agent_summary = f"{len(agent_list)}"

    save_video(frames, filename=None, fps=StandardizedEnvironment.ANIMATION_FPS, 
               scenario_type=scenario_type, agent_summary=agent_summary)
    # Also save as GIF
    save_gif_standardized(agent_list, r_min, filename=None, fps=StandardizedEnvironment.ANIMATION_FPS, 
                         num_moving_agents=num_moving_agents,
                         scenario_type=scenario_type, agent_summary=agent_summary)
    plt.close(fig)

def setup_standardized_scenario(env_type):
    """Set up scenario using standardized environment configuration."""
    print(f"Setting up {env_type.capitalize()} Environment using standardized configuration...")
    
    if env_type == 'doorway':
        obstacles = StandardizedEnvironment.get_doorway_obstacles()
    elif env_type == 'hallway':
        obstacles = StandardizedEnvironment.get_hallway_obstacles()
    elif env_type == 'intersection':
        obstacles = StandardizedEnvironment.get_intersection_obstacles()
    else:
        obstacles = []
    
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

    # Use standardized scenario setup
    obstacle_agents_x, obstacle_agents_v, obstacle_agents_target = setup_standardized_scenario(env_type)
    
    # --- Get User Input for Simulation ---

    # Get parameters for the moving drones
    num_moving_drones = get_input("Enter number of moving drones", 2, int)
    
    # Get simulation parameters from user
    min_radius = get_input("Enter minimum distance between drones", StandardizedEnvironment.DEFAULT_COLLISION_DISTANCE, float)
    
    # Add configurable wall collision distance
    wall_collision_multiplier = get_input("Enter wall collision distance multiplier (1.5-3.0 recommended)", 2.0, float)
    
    epsilon = get_input("Enter epsilon value", 0.1, float)
    step_size = get_input("Enter step size", 0.1, float)
    k_value = get_input("Enter k value", 10, int)
    max_steps = get_input("Enter maximum number of steps", 100, int)
    
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
    
    # Get drone positions using standardized positions
    ini_x_moving = []
    target_moving = []
    
    # Get standardized default positions
    standard_positions = StandardizedEnvironment.get_standard_agent_positions(env_type, num_moving_drones)
    
    for i in range(num_moving_drones):
        print(f"\n--- Agent {i+1} Parameters ---")
        
        # Get default values for this drone
        defaults = standard_positions[i] if i < len(standard_positions) else standard_positions[0]
        
        # Get start position
        start_x = get_input(f"Start X position (default: {defaults['start'][0]})", defaults['start'][0], float)
        start_y = get_input(f"Start Y position (default: {defaults['start'][1]})", defaults['start'][1], float)
        
        # Get goal position  
        goal_x = get_input(f"Goal X position (default: {defaults['goal'][0]})", defaults['goal'][0], float)
        goal_y = get_input(f"Goal Y position (default: {defaults['goal'][1]})", defaults['goal'][1], float)
        
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
        generate_animation_standardized(agent_list, min_radius, num_moving_agents=num_moving_drones, scenario_type=env_type, agent_summary=str(num_moving_drones))
    else:
        print("\nSimulation failed to find a solution.")

if __name__ == "__main__":
    main() 