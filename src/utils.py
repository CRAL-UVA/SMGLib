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