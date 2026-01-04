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
import matplotlib.lines as mlines
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
	"""Create animation from trajectory data using Social-CADRL styling.
	- Figure size: (12, 10)
	- Grid: True
	- Aspect: equal
	- Agent colors: consistent palette
	- Legend: outside, with Agents, Goals, Obstacles
	"""
	if not agents_data:
		print("No trajectory data available for animation")
		return None
	
	# Create animations directory
	animations_dir = output_dir / "animations"
	animations_dir.mkdir(exist_ok=True)
	
	# Set up the plot with CADRL-like styling
	fig, ax = plt.subplots(figsize=(12, 10))
	ax.set_aspect('equal')
	ax.grid(True)
	
	# Determine plot bounds from data, with margin
	all_x = []
	all_y = []
	for agent in agents_data:
		all_x.extend(agent.get('x', []))
		all_y.extend(agent.get('y', []))
	
	if all_x and all_y:
		margin = 2
		ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
		ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
	else:
		# Fallback bounds similar to CADRL defaults
		ax.set_xlim(-10, 10)
		ax.set_ylim(-10, 10)
	
	# Define consistent agent color palette
	agent_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
	
	# Scatter for agents (filled circles with black edge)
	agent_scatter = ax.scatter([], [], c=[], s=200, edgecolors='black', linewidths=1, label='Agents')
	
	# Obstacles from config (render as solid black filled shapes for uniform look)
	if config_file and os.path.exists(config_file):
		try:
			tree = ET.parse(config_file)
			root = tree.getroot()
			# ORCA-style polygon obstacles
			polys = root.findall('.//obstacle')
			for poly in polys:
				# Try vertex-based polygon
				vertices = []
				for vertex in poly.findall('vertex'):
					xr = vertex.get('xr')
					yr = vertex.get('yr')
					if xr is not None and yr is not None:
						vertices.append([float(xr), float(yr)])
				if len(vertices) >= 3:
					polygon = patches.Polygon(np.array(vertices), closed=True, facecolor='black', edgecolor='none', alpha=0.8)
					ax.add_patch(polygon)
				else:
					# Try rectangle style obstacles if present
					x1 = poly.find('x1')
					y1 = poly.find('y1')
					x2 = poly.find('x2')
					y2 = poly.find('y2')
					if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
						width = abs(float(x2.text) - float(x1.text))
						height = abs(float(y2.text) - float(y1.text))
						rect = patches.Rectangle((min(float(x1.text), float(x2.text)), min(float(y1.text), float(y2.text))),
												 width, height, linewidth=0, edgecolor='none', facecolor='black', alpha=0.8)
						ax.add_patch(rect)
		except Exception:
			pass
	
	# Goals for each agent (colored star with matching color)
	# Also collect legend handles
	legend_handles = []
	legend_labels = []
	
	# Agent legend entries
	if len(agents_data) > 1:
		for i, _ in enumerate(agents_data):
			color = agent_colors[i % len(agent_colors)]
			h = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
						 markersize=10, markerfacecolor=color, markeredgecolor='black')
			legend_handles.append(h)
			legend_labels.append(f'Agent {i+1}')
	else:
		h = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
					   markersize=10, markerfacecolor='blue', markeredgecolor='black')
		legend_handles.append(h)
		legend_labels.append('Agent')
	
	# Obstacle legend entry
	ob_h = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
					   markersize=10, markerfacecolor='black', markeredgecolor='none')
	legend_handles.append(ob_h)
	legend_labels.append('Obstacle')
	
	# Plot and legend entries for goals
	for i, agent in enumerate(agents_data):
		color = agent_colors[i % len(agent_colors)]
		goal = agent.get('goal_pos')
		if goal is not None:
			ax.plot(goal[0], goal[1], '*', color=color, markersize=15)
			h = mlines.Line2D([], [], color=color, marker='*', linestyle='None',
						   markersize=12, markerfacecolor=color, markeredgecolor='none')
			legend_handles.append(h)
			legend_labels.append(f'Goal {i+1}' if len(agents_data) > 1 else 'Goal')
	
	# Place legend outside the plot, match CADRL layout
	ax.legend(
		legend_handles, legend_labels,
		loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12, borderaxespad=0., markerscale=1.2
	)
	plt.tight_layout()
	plt.subplots_adjust(right=0.8)
	
	# Animation update function
	def animate(frame: int):
		positions = []
		colors = []
		for i, agent in enumerate(agents_data):
			x_list = agent.get('x', [])
			y_list = agent.get('y', [])
			if frame < len(x_list) and frame < len(y_list):
				positions.append([x_list[frame], y_list[frame]])
				colors.append(agent_colors[i % len(agent_colors)])
		if positions:
			agent_scatter.set_offsets(np.array(positions).reshape(-1, 2))
			agent_scatter.set_color(colors)
		else:
			agent_scatter.set_offsets(np.empty((0, 2)))
			agent_scatter.set_color([])
		return [agent_scatter]
	
	# Determine number of frames
	max_frames = max(len(agent.get('x', [])) for agent in agents_data)
	
	# Create animation using CADRL-like timing
	anim = FuncAnimation(fig, animate, frames=max_frames, interval=int(max(time_step, 0.05) * 1000 * 3), blit=True)
	
	# Save animation
	try:
		gif_path = animations_dir / "robot_movement.gif"
		anim.save(gif_path, writer='pillow', fps=round(1000 / max(1, int(max(time_step, 0.05) * 1000 * 3))))
		print(f"Animation saved to {gif_path}")
		saved_path = gif_path
	except Exception as e:
		print(f"Could not save GIF: {e}")
		try:
			html_path = animations_dir / "robot_movement.html"
			anim.save(html_path, writer='html')
			print(f"Animation saved to {html_path}")
			saved_path = html_path
		except Exception as e2:
			print(f"Could not save HTML either: {e2}")
			saved_path = None
	
	plt.close(fig)
	return saved_path

def calculate_oscillation_statistics(velocity, acceleration, heading, dt=0.1, weights={'v_zcr': 0.25,'jerk_energy': 0.25,'speed_ripple': 0.25,'heading_osc': 0.25}):
    n = len(velocity)
    if n < 5:
        raise ValueError("Input arrays too short for oscillation metrics")

    # Velocity Zero-Crossing Rate (ZCR) - normalized by a typical threshold
    acc_sign = np.sign(acceleration)
    zc = np.sum(np.abs(np.diff(acc_sign)) > 0)
    velocity_zero_cross_rate = zc / (n * dt)  # crossings per second
    # Normalize: typical smooth motion has 0-2 crossings/s, oscillatory motion has >5
    v_zcr_normalized = min(velocity_zero_cross_rate / 10.0, 1.0)  # scale to 0-1

    # Jerk Energy - use RMS jerk, normalized
    jerk = np.diff(acceleration) / dt
    jerk_rms = np.sqrt(np.mean(jerk ** 2))  # RMS jerk
    # Normalize: typical smooth motion has jerk < 1, oscillatory > 3
    jerk_normalized = min(jerk_rms / 5.0, 1.0)  # scale to 0-1

    # Speed Ripple Index (SRI) - already normalized (coefficient of variation)
    mean_v = np.mean(velocity)
    speed_ripple_index = np.std(velocity) / (mean_v + 1e-6)
    # Cap at 1.0 for very oscillatory motion
    sri_normalized = min(speed_ripple_index, 1.0)

    # Heading Oscillation Index (HOI) - normalize by typical values
    dtheta = np.diff(heading)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    ddtheta = np.diff(dtheta)
    heading_oscillation = np.mean(np.abs(ddtheta)) / dt  # rad/s^2
    # Normalize: smooth motion < 0.5 rad/s^2, oscillatory > 2 rad/s^2
    hoi_normalized = min(heading_oscillation / 2.0, 1.0)  # scale to 0-1

    components = {
        'v_zcr': velocity_zero_cross_rate,
        'jerk_energy': jerk_rms,
        'speed_ripple': speed_ripple_index,
        'heading_osc': heading_oscillation
    }

    # Weighted sum of normalized components (result between 0-1)
    osc_score = (
        weights['v_zcr'] * v_zcr_normalized +
        weights['jerk_energy'] * jerk_normalized +
        weights['speed_ripple'] * sri_normalized +
        weights['heading_osc'] * hoi_normalized
    )

    return {
        'velocity_zero_cross_rate': float(velocity_zero_cross_rate),
        'jerk_energy': float(jerk_rms),
        'speed_ripple_index': float(speed_ripple_index),
        'heading_oscillation_index': float(heading_oscillation),
        'oscillation_score': float(osc_score)
    }


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

# Standardized Environment Configuration
# This module provides consistent environment layouts and visualization parameters
# across all social navigation methods (CADRL, IMPC-DR, ORCA)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.lines as mlines

class StandardizedEnvironment:
    """
    Standardized environment configuration for social navigation methods.
    Provides consistent grid sizes, obstacle layouts, and visualization parameters.
    """
    
    # Standard grid dimensions (based on CADRL's larger grid for better visibility)
    GRID_X_MIN = -6.0
    GRID_X_MAX = 6.0
    GRID_Y_MIN = -8.0
    GRID_Y_MAX = 8.0
    
    # Standard agent parameters
    DEFAULT_AGENT_RADIUS = 0.3  # Reduced from 0.5 to allow closer passing
    DEFAULT_PREF_SPEED = 1.0
    DEFAULT_COLLISION_DISTANCE = 0.2  # Reduced from 0.3 to allow closer passing
    
    # Standard colors for agents (matching CADRL's color scheme)
    AGENT_COLORS = [
        [0.8500, 0.3250, 0.0980],  # orange
        [0.0, 0.4470, 0.7410],     # blue
        [0.4660, 0.6740, 0.1880],  # green
        [0.4940, 0.1840, 0.5560],  # purple
        [0.9290, 0.6940, 0.1250],  # yellow
        [0.3010, 0.7450, 0.9330],  # cyan
        [0.6350, 0.0780, 0.1840],  # chocolate
    ]
    
    # Standard visualization parameters
    FIG_SIZE = (10, 8)
    ANIMATION_FPS = 5
    ANIMATION_INTERVAL = 150  # milliseconds
    
    @classmethod
    def get_doorway_obstacles(cls):
        """
        Get standardized doorway obstacle layout.
        Vertical wall at x=0 with gap from y=-2 to y=2.
        """
        wall_x = 0.0
        gap_start = -2.0
        gap_end = 2.0
        
        # Create wall segments above and below the gap
        obstacle_ys = np.concatenate([
            np.linspace(cls.GRID_Y_MIN, gap_start, 12),
            np.linspace(gap_end, cls.GRID_Y_MAX, 12)
        ])
        
        obstacles = [np.array([wall_x, y]) for y in obstacle_ys]
        return obstacles
    
    @classmethod
    def get_hallway_obstacles(cls):
        """
        Get standardized hallway obstacle layout.
        Horizontal walls at y=-2 and y=2.
        Optimized for performance with fewer obstacles.
        """
        bottom_wall_y = -2.0
        top_wall_y = 2.0
        
        # Create bottom wall - reduced density for performance
        bottom_wall_xs = np.linspace(cls.GRID_X_MIN, cls.GRID_X_MAX, 12)  # Reduced from 20 to 12
        bottom_wall_positions = [np.array([x, bottom_wall_y]) for x in bottom_wall_xs]
        
        # Create top wall - reduced density for performance
        top_wall_xs = np.linspace(cls.GRID_X_MIN, cls.GRID_X_MAX, 12)  # Reduced from 20 to 12
        top_wall_positions = [np.array([x, top_wall_y]) for x in top_wall_xs]
        
        return bottom_wall_positions + top_wall_positions
    
    @classmethod
    def get_intersection_obstacles(cls):
        """
        Get standardized intersection obstacle layout.
        + shaped intersection with corridors centered at (0,0).
        Optimized for performance with fewer obstacles.
        """
        corridor_center = 0.0
        corridor_half_width = 2.0
        
        walls = []
        
        # Horizontal corridor walls - reduced density for performance
        for x in np.linspace(cls.GRID_X_MIN, cls.GRID_X_MAX, 12):  # Reduced from 25 to 12
            # Bottom wall of horizontal corridor
            walls.append(np.array([x, corridor_center - corridor_half_width]))
            # Top wall of horizontal corridor
            walls.append(np.array([x, corridor_center + corridor_half_width]))
        
        # Vertical corridor walls - reduced density for performance
        for y in np.linspace(cls.GRID_Y_MIN, cls.GRID_Y_MAX, 12):  # Reduced from 25 to 12
            # Left wall of vertical corridor
            walls.append(np.array([corridor_center - corridor_half_width, y]))
            # Right wall of vertical corridor
            walls.append(np.array([corridor_center + corridor_half_width, y]))
        
        # Remove walls in the intersection area itself
        filtered_walls = []
        intersection_min = corridor_center - corridor_half_width
        intersection_max = corridor_center + corridor_half_width
        
        for wall_pos in walls:
            x, y = wall_pos
            # Keep wall if it's not in the central intersection area
            if not (intersection_min <= x <= intersection_max and 
                    intersection_min <= y <= intersection_max):
                filtered_walls.append(wall_pos)
        
        return filtered_walls
    
    @classmethod
    def get_standard_agent_positions(cls, env_type, num_agents=2):
        """
        Get standardized agent start and goal positions for each environment type.
        """
        if env_type == 'doorway':
            # Agents moving through doorway - offset slightly to avoid head-on collision
            positions = [
                {'start': [-3.0, -0.5], 'goal': [3.0, 0.5]},  # Lower path
                {'start': [3.0, 0.5], 'goal': [-3.0, -0.5]}   # Upper path
            ]
        elif env_type == 'hallway':
            # Agents moving along hallway - offset to avoid head-on collision
            corridor_center = 0.0
            positions = [
                {'start': [-4.0, corridor_center - 0.3], 'goal': [4.0, corridor_center + 0.3]},  # Lower path
                {'start': [4.0, corridor_center + 0.3], 'goal': [-4.0, corridor_center - 0.3]}   # Upper path
            ]
        elif env_type == 'intersection':
            # Agents moving through intersection - offset to avoid collision at center
            corridor_center = 0.0
            positions = [
                {'start': [-4.0, corridor_center - 0.5], 'goal': [4.0, corridor_center + 0.5]},  # Offset horizontal
                {'start': [corridor_center - 0.5, -4.0], 'goal': [corridor_center + 0.5, 4.0]}   # Offset vertical
            ]
        else:
            # Generic positions
            positions = [
                {'start': [-3.0, -3.0], 'goal': [3.0, 3.0]},
                {'start': [3.0, 3.0], 'goal': [-3.0, -3.0]}
            ]
        
        return positions[:num_agents]
    
    @classmethod
    def create_standard_plot(cls, env_type, show_obstacles=True):
        """
        Create a standardized matplotlib plot with proper grid and obstacles.
        """
        fig, ax = plt.subplots(figsize=cls.FIG_SIZE)
        
        # Set grid limits
        ax.set_xlim(cls.GRID_X_MIN, cls.GRID_X_MAX)
        ax.set_ylim(cls.GRID_Y_MIN, cls.GRID_Y_MAX)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add obstacles if requested
        if show_obstacles:
            if env_type == 'doorway':
                obstacles = cls.get_doorway_obstacles()
            elif env_type == 'hallway':
                obstacles = cls.get_hallway_obstacles()
            elif env_type == 'intersection':
                obstacles = cls.get_intersection_obstacles()
            else:
                obstacles = []
            
            # Plot obstacles as gray circles
            for obs in obstacles:
                circle = patches.Circle(obs, radius=cls.DEFAULT_AGENT_RADIUS, 
                                      facecolor='gray', edgecolor='black', alpha=0.7)
                ax.add_patch(circle)
        
        # Set labels
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        
        return fig, ax
    
    @classmethod
    def create_standard_legend(cls, num_agents):
        """
        Create standardized legend matching CADRL's style.
        """
        legend_handles = []
        legend_labels = []
        
        # Obstacle legend
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                          markersize=10, markerfacecolor='gray', markeredgecolor='black'))
        legend_labels.append('Obstacle')
        
        # Agent legends
        for i in range(num_agents):
            color = cls.AGENT_COLORS[i % len(cls.AGENT_COLORS)]
            legend_handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                              markersize=10, markerfacecolor=color, markeredgecolor='black'))
            legend_labels.append(f'Agent {i+1}')
        
        # Goal legend
        legend_handles.append(mlines.Line2D([], [], color='green', marker='*', linestyle='None',
                                          markersize=12, markerfacecolor='green', markeredgecolor='none'))
        legend_labels.append('Goal')
        
        return legend_handles, legend_labels
    
    @classmethod
    def plot_agent_trajectory(cls, ax, positions, agent_id, show_trajectory=True, show_timestamps=True):
        """
        Plot agent trajectory with standardized styling.
        """
        if len(positions) == 0:
            return
        
        color = cls.AGENT_COLORS[agent_id % len(cls.AGENT_COLORS)]
        
        # Plot trajectory line
        if show_trajectory and len(positions) > 1:
            positions_array = np.array(positions)
            ax.plot(positions_array[:, 0], positions_array[:, 1], 
                   color=color, linewidth=2, alpha=0.7)
        
        # Plot start position as square
        start_pos = positions[0]
        ax.scatter(start_pos[0], start_pos[1], marker='s', s=200, 
                  edgecolor='black', facecolor=color, zorder=3)
        
        # Plot current position as circle
        current_pos = positions[-1]
        circle = patches.Circle(current_pos, radius=cls.DEFAULT_AGENT_RADIUS,
                              facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(circle)
        
        # Add timestamps if requested
        if show_timestamps and len(positions) > 5:
            # Add timestamp every few steps
            for i in range(0, len(positions), max(1, len(positions)//5)):
                pos = positions[i]
                ax.text(pos[0] - 0.15, pos[1] + 0.1, f'{i}', 
                       color=color, fontsize=8, alpha=0.7)
    
    @classmethod
    def plot_goal(cls, ax, goal_pos, agent_id):
        """
        Plot goal position with standardized styling.
        """
        color = cls.AGENT_COLORS[agent_id % len(cls.AGENT_COLORS)]
        ax.scatter(goal_pos[0], goal_pos[1], marker='*', s=300,
                  edgecolor='black', facecolor=color, zorder=4)

# Convenience functions for backward compatibility
def get_standardized_obstacles(env_type):
    """Get obstacles for the specified environment type."""
    if env_type == 'doorway':
        return StandardizedEnvironment.get_doorway_obstacles()
    elif env_type == 'hallway':
        return StandardizedEnvironment.get_hallway_obstacles()
    elif env_type == 'intersection':
        return StandardizedEnvironment.get_intersection_obstacles()
    else:
        return []

def get_standardized_positions(env_type, num_agents=2):
    """Get standard agent positions for the specified environment type."""
    return StandardizedEnvironment.get_standard_agent_positions(env_type, num_agents)

def create_standardized_plot(env_type, show_obstacles=True):
	"""Create a standardized plot for the specified environment type."""
	return StandardizedEnvironment.create_standard_plot(env_type, show_obstacles)


def calculate_fairness(agents_data, phi=None):
	num_agents = len(agents_data)
	if num_agents == 0:
		return 0.0
	
	if phi is None:
		phi = [1.0] * num_agents  # Equal weights
	
	# Ensure phi has the same length as agents_data
	if len(phi) < num_agents:
		# Pad with 1.0 for missing priorities
		phi = phi + [1.0] * (num_agents - len(phi))
	elif len(phi) > num_agents:
		phi = phi[:num_agents]

	R_local = [agent.get('path_deviation', 0.0) for agent in agents_data]

	# Calculate global social welfare 
	global_welfare = sum(phi[i] * R_local[i] for i in range(num_agents))

	return global_welfare