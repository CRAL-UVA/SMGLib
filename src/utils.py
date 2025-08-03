"""
Utility functions for SMGLib simulations.
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.spatial.distance import directed_hausdorff
from typing import Tuple, List, Dict, Any

def get_venv_python(base_dir: Path) -> str:
    """Get the path to the virtual environment Python executable."""
    venv_dir = base_dir / "venv"
    if os.name == 'nt':  # Windows
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")

def calculate_nominal_path(start_pos: Tuple[float, float], 
                          goal_pos: Tuple[float, float], 
                          num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the nominal path (straight line) from start to goal."""
    x = np.linspace(start_pos[0], goal_pos[0], num_steps)
    y = np.linspace(start_pos[1], goal_pos[1], num_steps)
    return x, y

def calculate_path_deviation(actual_x: List[float], actual_y: List[float],
                           nominal_x: np.ndarray, nominal_y: np.ndarray) -> Dict[str, float]:
    """Calculate path deviation metrics."""
    if len(actual_x) == 0 or len(nominal_x) == 0:
        return {
            'average_deviation': float('inf'),
            'max_deviation': float('inf'),
            'final_deviation': float('inf'),
            'hausdorff_distance': float('inf')
        }
    
    # Ensure same length for comparison
    min_len = min(len(actual_x), len(nominal_x))
    actual_x = actual_x[:min_len]
    actual_y = actual_y[:min_len]
    nominal_x = nominal_x[:min_len]
    nominal_y = nominal_y[:min_len]
    
    # Calculate point-wise deviations
    deviations = []
    for i in range(min_len):
        deviation = np.sqrt((actual_x[i] - nominal_x[i])**2 + (actual_y[i] - nominal_y[i])**2)
        deviations.append(deviation)
    
    # Calculate metrics
    average_deviation = np.mean(deviations) if deviations else float('inf')
    max_deviation = np.max(deviations) if deviations else float('inf')
    final_deviation = deviations[-1] if deviations else float('inf')
    
    # Hausdorff distance
    actual_points = np.column_stack((actual_x, actual_y))
    nominal_points = np.column_stack((nominal_x, nominal_y))
    hausdorff_distance = max(
        directed_hausdorff(actual_points, nominal_points)[0],
        directed_hausdorff(nominal_points, actual_points)[0]
    )
    
    return {
        'average_deviation': average_deviation,
        'max_deviation': max_deviation, 
        'final_deviation': final_deviation,
        'hausdorff_distance': hausdorff_distance
    }

def create_animation(agents_data: List[Dict], output_dir: Path, 
                    config_file: str = None, time_step: float = 0.1) -> Path:
    """Create animation from trajectory data."""
    if not agents_data:
        print("No trajectory data available for animation")
        return None
    
    # Create animations directory
    animations_dir = output_dir / "animations"
    animations_dir.mkdir(exist_ok=True)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Determine plot bounds
    all_x = []
    all_y = []
    for agent in agents_data:
        all_x.extend(agent['x'])
        all_y.extend(agent['y'])
    
    if all_x and all_y:
        margin = 2
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    # Load obstacles if config file exists
    if config_file and os.path.exists(config_file):
        tree = ET.parse(config_file)
        root = tree.getroot()
        obstacles = root.findall('.//obstacle')
        
        for obstacle in obstacles:
            try:
                x1 = float(obstacle.find('x1').text)
                y1 = float(obstacle.find('y1').text)
                x2 = float(obstacle.find('x2').text)
                y2 = float(obstacle.find('y2').text)
                
                # Draw obstacle as rectangle
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                rect = patches.Rectangle((min(x1, x2), min(y1, y2)), width, height,
                                       linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
                ax.add_patch(rect)
            except (AttributeError, ValueError, TypeError):
                continue
    
    # Initialize robot plots
    robot_plots = []
    trail_plots = []
    goal_plots = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(agents_data)))
    
    for i, agent in enumerate(agents_data):
        # Robot position
        robot_plot, = ax.plot([], [], 'o', markersize=10, color=colors[i], label=f'Robot {agent["id"]}')
        robot_plots.append(robot_plot)
        
        # Trail
        trail_plot, = ax.plot([], [], '-', alpha=0.5, color=colors[i])
        trail_plots.append(trail_plot)
        
        # Goal position
        goal_plot, = ax.plot(agent['goal_pos'][0], agent['goal_pos'][1], 's', 
                           markersize=8, color=colors[i], alpha=0.7)
        goal_plots.append(goal_plot)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position') 
    ax.set_title('Robot Navigation Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Animation function
    def animate(frame):
        for i, agent in enumerate(agents_data):
            if frame < len(agent['x']):
                # Update robot position
                robot_plots[i].set_data([agent['x'][frame]], [agent['y'][frame]])
                
                # Update trail
                trail_plots[i].set_data(agent['x'][:frame+1], agent['y'][:frame+1])
        
        return robot_plots + trail_plots
    
    # Determine number of frames
    max_frames = max(len(agent['x']) for agent in agents_data) if agents_data else 1
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=max_frames, interval=int(time_step*1000), 
                        blit=True, repeat=True)
    
    # Save animation
    try:
        anim.save(animations_dir / "robot_movement.gif", writer='pillow', fps=5)
        print(f"Animation saved to {animations_dir / 'robot_movement.gif'}")
    except Exception as e:
        print(f"Could not save GIF: {e}")
        try:
            anim.save(animations_dir / "robot_movement.html", writer='html')
            print(f"Animation saved to {animations_dir / 'robot_movement.html'}")
        except Exception as e2:
            print(f"Could not save HTML either: {e2}")
    
    plt.close(fig)
    return animations_dir / "robot_movement.gif"

def calculate_makespan_ratios(completion_times: List[float]) -> List[float]:
    """Calculate makespan ratios for agents."""
    if not completion_times:
        return []
    
    # Filter out infinite/invalid times
    valid_times = [t for t in completion_times if t != float('inf') and t > 0]
    
    if not valid_times:
        return [float('inf')] * len(completion_times)
    
    fastest_time = min(valid_times)
    ratios = []
    
    for time in completion_times:
        if time == float('inf') or time <= 0:
            ratios.append(float('inf'))
        else:
            ratios.append(time / fastest_time)
    
    return ratios

def print_simulation_results(method_name: str, num_robots: int, makespan: float, 
                           flow_rate: float, completion_data: List[Dict]):
    """Print formatted simulation results."""
    print(f"\n{'='*60}")
    print(f"{method_name.upper()} SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Number of robots: {num_robots}")
    print(f"Makespan: {makespan:.3f} seconds")
    print(f"Flow Rate: {flow_rate:.4f} agents/(unit·s)")
    
    # Agent completion summary
    successful_agents = sum(1 for agent in completion_data if agent.get('reached_goal', False))
    print(f"Agents that reached goals: {successful_agents}/{len(completion_data)}")
    
    # Individual agent results
    completion_times = [agent.get('completion_time', float('inf')) for agent in completion_data]
    makespan_ratios = calculate_makespan_ratios(completion_times)
    
    print(f"\nMakespan Ratios (MR_i = TTG_i / TTG_fastest):")
    for i, (agent, mr) in enumerate(zip(completion_data, makespan_ratios)):
        ttg = agent.get('completion_time', float('inf'))
        if ttg != float('inf'):
            print(f"Robot {agent.get('id', i)}: TTG = {ttg:.3f}s, MR = {mr:.3f} ✓")
        else:
            print(f"Robot {agent.get('id', i)}: TTG = ∞, MR = N/A ✗ (did not reach goal)")
    
    print(f"{'='*60}")

def save_trajectory_csv(agents_data: List[Dict], output_dir: Path) -> Path:
    """Save trajectory data to CSV files."""
    velocity_csv = output_dir / "velocities.csv"
    
    # Create velocity CSV
    with open(velocity_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['robot_id', 'time', 'x', 'y', 'vx', 'vy'])
        
        for agent in agents_data:
            robot_id = agent['id']
            positions = list(zip(agent['x'], agent['y']))
            velocities = agent.get('velocities', [])
            
            for i, (pos, vel) in enumerate(zip(positions, velocities)):
                time = i * 0.1  # Assuming 0.1s time step
                writer.writerow([robot_id, time, pos[0], pos[1], vel[0], vel[1]])
    
    # Create individual robot trajectory files
    for agent in agents_data:
        robot_csv = output_dir / f"robot_{agent['id']}_trajectory.csv"
        with open(robot_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'y', 'vx', 'vy'])
            
            velocities = agent.get('velocities', [])
            for i, (x, y) in enumerate(zip(agent['x'], agent['y'])):
                time = i * 0.1
                vx = velocities[i][0] if i < len(velocities) else 0
                vy = velocities[i][1] if i < len(velocities) else 0
                writer.writerow([time, x, y, vx, vy])
    
    return velocity_csv 