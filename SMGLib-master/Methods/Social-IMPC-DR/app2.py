import numpy as np
import sys
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from test import PLAN
from plot import plot_trajectory

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

def generate_animation(agent_list, r_min, filename="video_recordings/simulation.avi"):
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
            color = colors[i % len(colors)]
            pos = agent.position[step]
            
            # Draw trajectory as a dotted line
            if step > 0:
                past_positions = np.array(agent.position[:step+1])
                ax.plot(past_positions[:, 0], past_positions[:, 1], linestyle="dotted", color=color, linewidth=2)
            
            # Draw solid line for completed part of the trajectory
            if step > 5:
                completed_positions = np.array(agent.position[max(0, step-5):step+1])
                ax.plot(completed_positions[:, 0], completed_positions[:, 1], linestyle="solid", color=color, linewidth=2)

            # Draw drone as a circle
            circle = Circle(pos, radius=r_min / 2.0, edgecolor='black', facecolor=color, zorder=3)
            ax.add_patch(circle)

            # Mark start position with a square
            ax.scatter(agent.position[0][0], agent.position[0][1], marker='s', s=100, edgecolor='black', color=color)

            # Mark target with a diamond
            ax.scatter(agent.target[0], agent.target[1], marker='d', s=100, edgecolor='black', color=color)
        
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(image)

    save_video(frames, filename)
    plt.close(fig)


def main():
    # Get simulation parameters
    num_drones = get_input("Enter number of drones", 2, int)
    min_radius = get_input("Enter minimum distance between drones", 0.5, float)
    epsilon = get_input("Enter epsilon value", 0.1, float)
    step_size = get_input("Enter step size", 0.1, float)
    k_value = get_input("Enter k value", 10, int)
    max_steps = get_input("Enter maximum number of steps", 100, int)
    random_init_targets = get_input("Use random initial and target positions? (y/n)", "n", str).lower() == "y"
    
    # Generate initial positions
    if random_init_targets:
        ini_x = [np.random.rand(2) * 2 for _ in range(num_drones)]
        ini_v = [np.zeros(2) for _ in range(num_drones)]
    else:
        ini_x = []
        ini_v = []
        for i in range(num_drones):
            print(f"\nDrone {i}:")
            pos = get_position_input(f"Enter initial position (x y): ")
            vel = get_position_input(f"Enter initial velocity (vx vy): ")
            ini_x.append(pos)
            ini_v.append(vel)
    
    # Generate targets
    if random_init_targets:
        target = [np.random.rand(2) * 2 for _ in range(num_drones)]
    else:
        target = []
        for i in range(num_drones):
            pos = get_position_input(f"Enter target position for drone {i} (x y): ")
            target.append(pos)
    
    print("\nStarting simulation...")
    result, agent_list = PLAN(num_drones, ini_x, ini_v, target, min_radius, epsilon, step_size, k_value, max_steps)
    
    if result:
        print("\nSimulation completed successfully!")
        generate_animation(agent_list, min_radius)
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
        print(f"Drone {i}: {agent.position[-1]}")
    print(f"Final velocities:")
    for i, agent in enumerate(agent_list):
        print(f"Drone {i}: {agent.v[-1]}")

if __name__ == "__main__":
    main()