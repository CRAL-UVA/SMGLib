import numpy as np
import sys
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from test import PLAN
from plot import plot_trajectory
import json

def get_input(prompt, default, type_cast=str):
    while True:
        user_input = input(f"{prompt} (default: {default}): ")
        if not user_input:
            return default
        try:
            return type_cast(user_input)
        except ValueError:
            print(f"Invalid input! Please enter a valid {type_cast.__name__}.")

def get_position_input(prompt):
    while True:
        try:
            user_input = input(prompt)
            x, y = map(float, user_input.split())
            return np.array([x, y])
        except ValueError:
            print("Invalid input! Please enter two numbers separated by space (e.g., '1.0 2.0')")

def save_video(frames, filename="video_recordings/simulation.avi", fps=5):
    if not frames:
        print("No frames captured. Cannot save video.")
        return
    
    if not os.path.exists("video_recordings"):
        os.makedirs("video_recordings")
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved as {filename}")

def save_gif(agent_list, r_min, filename="video_recordings/simulation.gif", fps=5, num_moving_agents=None):
    """Save animation as GIF file."""
    if not os.path.exists("video_recordings"):
        os.makedirs("video_recordings")
    
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["orange", "blue"]  # Color assignment for two drones
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
        for i, agent in enumerate(agent_list):
            if num_moving_agents is not None:
                if i < num_moving_agents:
                    color = colors[i % len(colors)]
                else:
                    color = 'gray'
            else:
                color = colors[i % len(colors)]
            
            # Use last known position if frame is out of bounds
            pos_index = min(frame, len(agent.position) - 1)
            pos = agent.position[pos_index]
            
            # Draw trajectory as a dotted line
            if pos_index > 0:
                past_positions = np.array(agent.position[:pos_index+1])
                ax.plot(past_positions[:, 0], past_positions[:, 1], linestyle="dotted", color=color, linewidth=2)
            
            # Draw solid line for completed part of the trajectory
            if pos_index > 5:
                completed_positions = np.array(agent.position[max(0, pos_index-5):pos_index+1])
                ax.plot(completed_positions[:, 0], completed_positions[:, 1], linestyle="solid", color=color, linewidth=2)

            # Draw drone as a circle
            circle = Circle(pos, radius=r_min / 2.0, edgecolor='black', facecolor=color, zorder=3)
            ax.add_patch(circle)

            # Mark start position with a square
            ax.scatter(agent.position[0][0], agent.position[0][1], marker='s', s=100, edgecolor='black', color=color)

            # Mark target with a diamond
            if num_moving_agents is None or i < num_moving_agents:
                ax.scatter(agent.target[0], agent.target[1], marker='d', s=100, edgecolor='black', color=color)
        
        return ax,
    
    anim = FuncAnimation(fig, animate, frames=len(agent_list[0].position), interval=1000//fps, blit=False)
    
    try:
        anim.save(filename, writer='pillow', fps=fps)
        print(f"GIF saved as {filename}")
    except Exception as e:
        print(f"Failed to save GIF: {e}")
        print("Saving as HTML instead...")
        try:
            anim.save(filename.replace('.gif', '.html'), writer='html')
            print(f"HTML animation saved as {filename.replace('.gif', '.html')}")
        except Exception as e2:
            print(f"Failed to save HTML: {e2}")
    
    plt.close(fig)

def generate_animation(agent_list, r_min, filename="video_recordings/simulation.avi", num_moving_agents=None):
    frames = []
    fig, ax = plt.subplots(figsize=(5, 5))
    
    colors = ["orange", "blue"]  # Color assignment for two drones
    
    for step in range(len(agent_list[0].position)):
        ax.clear()
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
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

            # Draw drone as a circle
            circle = Circle(pos, radius=r_min / 2.0, edgecolor='black', facecolor=color, zorder=3)
            ax.add_patch(circle)

            # Mark start position with a square
            ax.scatter(agent.position[0][0], agent.position[0][1], marker='s', s=100, edgecolor='black', color=color)

            # Mark target with a diamond
            if num_moving_agents is None or i < num_moving_agents:
                ax.scatter(agent.target[0], agent.target[1], marker='d', s=100, edgecolor='black', color=color)
        
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(image)

    save_video(frames, filename)
    # Also save as GIF
    save_gif(agent_list, r_min, filename.replace('.avi', '.gif'), num_moving_agents=num_moving_agents)
    plt.close(fig)

def setup_doorway_scenario():
    """Sets up the stationary wall for the doorway scenario."""
    print("Setting up Doorway Environment...")
    
    # --- Stationary Obstacle Drones (The Wall) ---
    wall_x = 1.0  # The wall is now vertical and centered
    gap_start = 0.8 # y-coordinate for the start of the gap
    gap_end = 1.6   # y-coordinate for the end of the gap. Gap width is now 0.8.
    
    # y-coordinates for the wall segments - increased density for better coverage
    # More obstacles to ensure continuous wall coverage
    obstacle_ys = np.concatenate([
        np.linspace(0.2, gap_start, 8),  # Increased from 4 to 8 obstacles
        np.linspace(gap_end, 2.3, 8)     # Increased from 4 to 8 obstacles
    ])
    
    ini_x_obstacles = [np.array([wall_x, y]) for y in obstacle_ys]
    ini_v_obstacles = [np.zeros(2) for _ in ini_x_obstacles]
    # Stationary agents have their target set to their start position
    target_obstacles = ini_x_obstacles 

    return ini_x_obstacles, ini_v_obstacles, target_obstacles

def setup_hallway_scenario():
    """Sets up the stationary walls for the hallway scenario."""
    print("Setting up Hallway Environment...")
    
    # --- Stationary Obstacle Drones (The Walls) ---
    # Create horizontal walls at top and bottom to form a narrower corridor
    # Bottom wall at y=0.7, Top wall at y=1.7 
    # This creates a corridor between y=0.7 and y=1.7 (width of 1.0 units - narrower but passable)
    
    bottom_wall_y = 0.7
    top_wall_y = 1.7
    
    # Create bottom wall (horizontal line of obstacles) - increased density
    bottom_wall_xs = np.linspace(0.1, 2.4, 15)  # Increased from 10 to 15 obstacles
    bottom_wall_positions = [np.array([x, bottom_wall_y]) for x in bottom_wall_xs]
    
    # Create top wall (horizontal line of obstacles) - increased density
    top_wall_xs = np.linspace(0.1, 2.4, 15)  # Increased from 10 to 15 obstacles
    top_wall_positions = [np.array([x, top_wall_y]) for x in top_wall_xs]
    
    # Combine all wall positions
    ini_x_obstacles = bottom_wall_positions + top_wall_positions
    ini_v_obstacles = [np.zeros(2) for _ in ini_x_obstacles]
    # Stationary agents have their target set to their start position
    target_obstacles = ini_x_obstacles
    
    return ini_x_obstacles, ini_v_obstacles, target_obstacles

def setup_intersection_scenario():
    """Sets up the stationary walls for the intersection scenario."""
    print("Setting up Intersection Environment...")
    
    # --- Stationary Obstacle Drones (The Walls) ---
    # Create walls defining the + shaped intersection corridors
    # The intersection opening is centered at (1.25, 1.25) with wider corridors
    
    corridor_center = 1.25
    corridor_half_width = 0.4  # Half-width of each corridor (gap from center to wall)
    
    # Wall positions - only the edges defining the corridors
    walls = []
    
    # Horizontal corridor walls (left-right passage) - increased density
    # Bottom wall of horizontal corridor
    for x in np.linspace(0.1, 2.4, 18):  # Increased from 12 to 18 obstacles
        y = corridor_center - corridor_half_width
        walls.append(np.array([x, y]))
    
    # Top wall of horizontal corridor  
    for x in np.linspace(0.1, 2.4, 18):  # Increased from 12 to 18 obstacles
        y = corridor_center + corridor_half_width
        walls.append(np.array([x, y]))
    
    # Vertical corridor walls (bottom-top passage) - increased density
    # Left wall of vertical corridor
    for y in np.linspace(0.1, 2.4, 18):  # Increased from 12 to 18 obstacles
        x = corridor_center - corridor_half_width
        walls.append(np.array([x, y]))
    
    # Right wall of vertical corridor
    for y in np.linspace(0.1, 2.4, 18):  # Increased from 12 to 18 obstacles
        x = corridor_center + corridor_half_width
        walls.append(np.array([x, y]))
    
    # Remove wall agents that are in the intersection area itself
    # (where horizontal and vertical corridors meet)
    filtered_walls = []
    intersection_min = corridor_center - corridor_half_width
    intersection_max = corridor_center + corridor_half_width
    
    for wall_pos in walls:
        x, y = wall_pos
        # Keep wall if it's not in the central intersection area
        if not (intersection_min <= x <= intersection_max and 
                intersection_min <= y <= intersection_max):
            filtered_walls.append(wall_pos)
    
    ini_x_obstacles = filtered_walls
    ini_v_obstacles = [np.zeros(2) for _ in ini_x_obstacles]
    # Stationary agents have their target set to their start position
    target_obstacles = ini_x_obstacles
    
    return ini_x_obstacles, ini_v_obstacles, target_obstacles

def main():
    env_type = None
    robot_configs = None
    # Parse command-line arguments for env_type and --robot_configs
    for arg in sys.argv[1:]:
        if arg.startswith('--robot_configs='):
            robot_configs = json.loads(arg.split('=', 1)[1])
        elif env_type is None:
            env_type = arg

    obstacle_agents_x = []
    obstacle_agents_v = []
    obstacle_agents_target = []
    
    if env_type == 'doorway':
        obstacle_agents_x, obstacle_agents_v, obstacle_agents_target = setup_doorway_scenario()
    elif env_type == 'hallway':
        obstacle_agents_x, obstacle_agents_v, obstacle_agents_target = setup_hallway_scenario()
    elif env_type == 'intersection':
        obstacle_agents_x, obstacle_agents_v, obstacle_agents_target = setup_intersection_scenario()
    
    # If robot_configs is provided, use those positions and skip prompts
    if robot_configs is not None:
        print("Using robot positions from unified user selector.")
        num_moving_drones = len(robot_configs)
        ini_x_moving = [np.array([r['start_x'], r['start_y']]) for r in robot_configs]
        target_moving = [np.array([r['goal_x'], r['goal_y']]) for r in robot_configs]
        # Use all default values for other parameters
    if env_type == 'hallway':
            min_radius = 0.08
        else:
            min_radius = 0.1
        wall_collision_multiplier = 2.0
        epsilon = 0.1
        step_size = 0.1
        k_value = 10
        max_steps = 100
    else:
        print("Error: This script must be called with robot_configs from the unified selector.")
        sys.exit(1)
    
    ini_v_moving = [np.zeros(2) for _ in range(num_moving_drones)]
    ini_x = ini_x_moving + obstacle_agents_x
    ini_v = ini_v_moving + obstacle_agents_v
    target = target_moving + obstacle_agents_target
    num_drones = len(ini_x)
    print("\nStarting simulation...")
    result, agent_list, completion_step = PLAN(num_drones, ini_x, ini_v, target, min_radius, epsilon, step_size, k_value, max_steps, num_moving_drones=num_moving_drones, wall_collision_multiplier=wall_collision_multiplier)
    # Save completion step for Flow Rate calculation
    with open("completion_step.txt", "w") as f:
        f.write(str(completion_step))
    if result:
        print("\nSimulation completed successfully!")
        generate_animation(agent_list, min_radius, num_moving_agents=num_moving_drones)
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