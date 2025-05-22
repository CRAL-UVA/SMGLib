#!/usr/bin/env python3
import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import csv
import numpy as np
from pathlib import Path
from scipy.spatial.distance import directed_hausdorff
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def get_venv_python():
    venv_dir = Path(__file__).parent / "venv"
    if sys.platform == "win32":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")

def calculate_nominal_path(start_pos, goal_pos, num_steps):
    """Calculate the nominal path (straight line) from start to goal."""
    x = np.linspace(start_pos[0], goal_pos[0], num_steps)
    y = np.linspace(start_pos[1], goal_pos[1], num_steps)
    return x, y

def parse_orca_log(log_file):
    tree = ET.parse(log_file)
    root = tree.getroot()
    
    # Extract simulation parameters
    agents_elem = root.find('agents')
    num_robots = int(agents_elem.get('number'))
    time_step = float(root.find('.//timestep').text)
    
    # Extract agent data from the log section
    log_section = root.find('log')
    agents_data = []
    
    # First get the initial agent definitions to get start and goal positions
    agent_defs = {}
    for agent in agents_elem.findall('agent'):
        agent_id = int(agent.get('id'))
        agent_defs[agent_id] = agent
    
    # Then process the log data
    for agent_log in log_section.findall('agent'):
        agent_id = int(agent_log.get('id'))
        agent_def = agent_defs[agent_id]
        
        positions = []
        velocities = []
        
        # Get start and goal positions from the initial agent definition
        start_pos = [float(agent_def.get('start.xr')), float(agent_def.get('start.yr'))]
        goal_pos = [float(agent_def.get('goal.xr')), float(agent_def.get('goal.yr'))]
        
        # Extract trajectory data from the log section
        path = agent_log.find('path')
        if path is not None:
            steps = path.findall('step')
            for i in range(len(steps)):
                step = steps[i]
                pos = [float(step.get('xr')), float(step.get('yr'))]
                positions.append(pos)
                
                # Calculate velocity from position difference
                if i < len(steps) - 1:
                    next_step = steps[i+1]
                    next_pos = [float(next_step.get('xr')), float(next_step.get('yr'))]
                    # Calculate velocity as (next_pos - pos) / time_step
                    vel = [(next_pos[0] - pos[0]) / time_step, (next_pos[1] - pos[1]) / time_step]
                else:
                    # For the last step, use zero velocity
                    vel = [0, 0]
                velocities.append(vel)
        
        agents_data.append({
            'id': agent_id,
            'positions': positions,
            'velocities': velocities,
            'start_pos': start_pos,
            'goal_pos': goal_pos
        })
    
    return num_robots, time_step, agents_data

def generate_orca_csvs(log_file, output_dir):
    num_robots, time_step, agents_data = parse_orca_log(log_file)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create velocity CSV file
    velocity_csv = output_dir / "velocities.csv"
    with open(velocity_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header with robot IDs
        header = ['step']
        for agent in agents_data:
            header.extend([f'robot_{agent["id"]}_vx', f'robot_{agent["id"]}_vy'])
        writer.writerow(header)
        
        # Write velocity data for each timestep
        max_steps = max(len(agent['velocities']) for agent in agents_data)
        for t in range(max_steps):
            row = [t]
            for agent in agents_data:
                if t < len(agent['velocities']):
                    vel = agent['velocities'][t]
                    row.extend([vel[0], vel[1]])
                else:
                    row.extend([0, 0])  # Pad with zeros if trajectory is shorter
            writer.writerow(row)
    
    # Generate trajectory CSV for each agent
    for agent in agents_data:
        num_steps = len(agent['positions'])
        nominal_x, nominal_y = calculate_nominal_path(agent['start_pos'], agent['goal_pos'], num_steps)
        
        # Create CSV with columns: x, y, nominal_x, nominal_y
        output_csv = output_dir / f"robot_{agent['id']}_trajectory.csv"
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y', 'nominal_x', 'nominal_y'])
            
            # Write data for each timestep
            for t in range(num_steps):
                pos = agent['positions'][t]
                writer.writerow([
                    pos[0],         # x
                    pos[1],         # y
                    nominal_x[t],   # nominal_x
                    nominal_y[t]    # nominal_y
                ])
    
    return velocity_csv

def evaluate_velocities(velocity_csv):
    """Evaluate the velocities using the evaluate module."""
    # Read the velocity CSV
    data = pd.read_csv(velocity_csv)
    
    # Process each robot's velocities
    robot_ids = []
    for col in data.columns:
        if col.endswith('_vx'):
            robot_id = col.split('_')[1]
            robot_ids.append(robot_id)
    
    # Calculate average delta velocity for each robot
    for robot_id in robot_ids:
        vx_col = f'robot_{robot_id}_vx'
        vy_col = f'robot_{robot_id}_vy'
        
        # Calculate resultant velocity
        data[f'robot_{robot_id}_resultant'] = np.sqrt(data[vx_col]**2 + data[vy_col]**2)
        
        # Calculate differences
        diffs = np.diff(data[f'robot_{robot_id}_resultant'])
        abs_diffs = np.abs(diffs)
        sum_abs_diffs = np.sum(abs_diffs)
        
        # Print the average delta velocity
        print("*" * 65)
        print(f"Robot {robot_id} Avg delta velocity: {sum_abs_diffs:.4f}")
        print("*" * 65)
    
    return sum_abs_diffs  # Return the last calculated value

def get_num_robots_from_config(config_file):
    """Extract number of robots from config file."""
    tree = ET.parse(config_file)
    root = tree.getroot()
    agents = root.findall('.//agent')
    return len(agents)

def generate_animation(agents_data, output_dir, map_size=(64, 64), config_file=None):
    """Generate an animation of robot movements."""
    # Create animations directory in the logs folder
    animations_dir = output_dir.parent / "animations"
    animations_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Create scatter plots for robots
    scatter = ax.scatter([], [], c='blue', s=100)
    
    # Create goal markers
    for agent in agents_data:
        goal_pos = agent['goal_pos']
        ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=10, label='Goal' if agent['id'] == 0 else "")
    
    # Add obstacles from config file if provided
    if config_file and os.path.exists(config_file):
        tree = ET.parse(config_file)
        root = tree.getroot()
        obstacles = root.findall('.//obstacle')
        
        for obstacle in obstacles:
            vertices = []
            for vertex in obstacle.findall('vertex'):
                x = float(vertex.get('xr'))
                y = float(vertex.get('yr'))
                vertices.append([x, y])
            
            # Create polygon patch for obstacle
            vertices = np.array(vertices)
            polygon = patches.Polygon(vertices, closed=True, facecolor='gray', alpha=0.5)
            ax.add_patch(polygon)
    
    # Create velocity vectors
    velocity_arrows = []
    for _ in agents_data:
        arrow = ax.arrow(0, 0, 0, 0, head_width=0.5, head_length=0.8, fc='red', ec='red', alpha=0.5)
        velocity_arrows.append(arrow)
    
    def update(frame):
        # Update robot positions
        positions = []
        for agent in agents_data:
            if frame < len(agent['positions']):
                pos = agent['positions'][frame]
                positions.append(pos)
                
                # Update velocity arrow
                if frame < len(agent['velocities']):
                    vel = agent['velocities'][frame]
                    arrow = velocity_arrows[agent['id']]
                    arrow.set_data(x=pos[0], y=pos[1], dx=vel[0], dy=vel[1])
            else:
                # If we're past the end of this agent's trajectory, use the last position
                positions.append(agent['positions'][-1])
        
        scatter.set_offsets(positions)
        return [scatter] + velocity_arrows
    
    # Create animation with reduced frames
    num_frames = max(len(agent['positions']) for agent in agents_data)
    # Sample every 5th frame to reduce total frames
    frames = range(0, num_frames, 5)
    anim = FuncAnimation(fig, update, frames=frames, interval=200, blit=True)  # 200ms interval for 5 FPS
    
    # Add legend
    ax.legend()
    
    # Save animation
    try:
        anim.save(animations_dir / "robot_movement.gif", writer='pillow', fps=5)  # Set FPS to 5
        print(f"Animation saved to {animations_dir / 'robot_movement.gif'}")
    except Exception as e:
        print(f"Failed to save GIF: {e}")
        print("Saving as HTML instead...")
        try:
            anim.save(animations_dir / "robot_movement.html", writer='html')
            print(f"Animation saved to {animations_dir / 'robot_movement.html'}")
        except Exception as e:
            print(f"Failed to save HTML: {e}")
            print("Saving individual frames as PNG...")
            for frame in range(num_frames):
                update(frame)
                plt.savefig(animations_dir / f"frame_{frame:04d}.png")
            print(f"Frames saved to {animations_dir}")
    
    plt.close()
    return animations_dir / "robot_movement.gif"

def run_social_orca(config_file, num_robots):
    print("\nRunning Social-ORCA Simulation")
    print("=============================")
    
    # Store the base directory
    base_dir = Path(__file__).parent
    
    # Change to Social-ORCA directory
    orca_dir = base_dir / "Methods/Social-ORCA"
    os.chdir(orca_dir)
    
    # Use the provided config file
    config_path = config_file
    print(f"\nUsing configuration file: {config_path}")
    
    try:
        num_robots = get_num_robots_from_config(config_path)
    except Exception as e:
        print(f"Error reading config file: {e}")
        return
    
    # Run the simulation
    print(f"\nRunning simulation with {num_robots} robots...")
    cmd = f"./build/single_test {config_path} {num_robots}"
    subprocess.run(cmd, shell=True)
    
    # Get the most recent log file
    log_files = sorted(Path("logs").glob("*_log.xml"), key=os.path.getctime)
    if not log_files:
        print("\nNo log file was generated!")
        return
    
    latest_log = log_files[-1]
    print(f"\nLog file generated: {latest_log}")
    
    # Generate CSV files in a new 'trajectories' directory
    output_dir = latest_log.parent / "trajectories"
    try:
        velocity_csv = generate_orca_csvs(latest_log, output_dir)
        print(f"\nTrajectory CSV files generated in: {output_dir}")
        
        # Generate animation
        num_robots, time_step, agents_data = parse_orca_log(latest_log)
        animation_path = generate_animation(agents_data, output_dir, config_file=config_path)
        print(f"\nAnimation generated at: {animation_path}")
        
        # Evaluate trajectories
        print("\nEvaluating trajectories...")
        trajectory_dir = output_dir
        trajectory_files = list(trajectory_dir.glob("robot_*_trajectory.csv"))
        
        for i, traj_file in enumerate(trajectory_files):
            print(f"\nEvaluating Robot {i} trajectory:")
            data = pd.read_csv(traj_file)
            
            # Extract coordinates
            actual_x, actual_y = data.iloc[:, 0], data.iloc[:, 1]
            nominal_x, nominal_y = data.iloc[:, 2], data.iloc[:, 3]
            
            # Compute trajectory difference and L2 norm
            diff_x, diff_y = actual_x - nominal_x, actual_y - nominal_y
            l2_norm = np.sqrt(diff_x**2 + diff_y**2).sum()
            
            # Calculate Hausdorff distance
            actual_trajectory = np.column_stack((actual_x, actual_y))
            nominal_trajectory = np.column_stack((nominal_x, nominal_y))
            hausdorff_dist = directed_hausdorff(actual_trajectory, nominal_trajectory)[0]
            
            print("*" * 65)
            print(f"Robot {i} Path Deviation Metrics:")
            print(f"L2 Norm: {l2_norm:.4f}")
            print(f"Hausdorff distance: {hausdorff_dist:.4f}")
            print("*" * 65)
        
        evaluate_velocities(velocity_csv)
    except Exception as e:
        print(f"Error processing trajectories: {e}")
        return

def run_social_impc_dr(env_type='doorway'):
    print("\nRunning Social-IMPC-DR Simulation")
    print("=================================")
    
    # Change to Social-IMPC-DR directory using absolute path
    impc_dir = Path(__file__).parent / "Methods" / "Social-IMPC-DR"
    if not impc_dir.exists():
        print(f"Error: Directory {impc_dir} not found!")
        return
    
    print(f"Changing to directory: {impc_dir}")
    os.chdir(impc_dir)
    
    # Run app2.py with environment parameter
    subprocess.run([get_venv_python(), "app2.py", env_type])
    
    # Evaluate trajectory if available
    path_deviation_files = list(impc_dir.glob("path_deviation_robot_*.csv"))
    if path_deviation_files:
        print("\nEvaluating Social-IMPC-DR trajectories:")
        for path_deviation_file in path_deviation_files:
            data = pd.read_csv(path_deviation_file)
            
            # Extract coordinates
            actual_x, actual_y = data.iloc[:, 0], data.iloc[:, 1]
            nominal_x, nominal_y = data.iloc[:, 2], data.iloc[:, 3]
            
            # Compute trajectory difference and L2 norm
            diff_x, diff_y = actual_x - nominal_x, actual_y - nominal_y
            l2_norm = np.sqrt(diff_x**2 + diff_y**2).sum()
            
            # Calculate Hausdorff distance
            actual_trajectory = np.column_stack((actual_x, actual_y))
            nominal_trajectory = np.column_stack((nominal_x, nominal_y))
            hausdorff_dist = directed_hausdorff(actual_trajectory, nominal_trajectory)[0]
            
            print("*" * 65)
            print(f"Robot {path_deviation_file.stem.split('_')[-1]} Path Deviation Metrics:")
            print(f"L2 Norm: {l2_norm:.4f}")
            print(f"Hausdorff distance: {hausdorff_dist:.4f}")
            print("*" * 65)
    else:
        print(f"\nWarning: No path deviation CSV files found in {impc_dir}")

def generate_config(env_type, num_robots, robot_positions):
    """Generate a configuration file for the simulation."""
    root = ET.Element('root')
    
    # Add agents section
    agents = ET.SubElement(root, 'agents', {'number': str(num_robots), 'type': 'orca'})
    default_params = ET.SubElement(agents, 'default_parameters', {
        'size': '0.3',
        'movespeed': '1',
        'agentsmaxnum': str(num_robots),
        'timeboundary': '5.4',
        'sightradius': '3.0',
        'timeboundaryobst': '33'
    })
    
    # Add individual agents
    for i in range(num_robots):
        agent = ET.SubElement(agents, 'agent', {
            'id': str(i),
            'start.xr': str(robot_positions[i]['start_x']),
            'start.yr': str(robot_positions[i]['start_y']),
            'goal.xr': str(robot_positions[i]['goal_x']),
            'goal.yr': str(robot_positions[i]['goal_y'])
        })
    
    # Add map section
    map_elem = ET.SubElement(root, 'map')
    ET.SubElement(map_elem, 'width').text = '64'
    ET.SubElement(map_elem, 'height').text = '64'
    ET.SubElement(map_elem, 'cellsize').text = '1'
    
    # Add grid
    grid = ET.SubElement(map_elem, 'grid')
    for _ in range(64):  # 64x64 grid
        row = ET.SubElement(grid, 'row')
        row.text = '0 ' * 63 + '0'  # 64 zeros per row
    
    # Add obstacles based on environment type
    if env_type == 'hallway':
        obstacles = ET.SubElement(root, 'obstacles', {'number': '2'})
        # Add hallway walls
        obstacle1 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle1, 'vertex', {'xr': '0', 'yr': '31'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '0', 'yr': '32'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '63', 'yr': '31'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '63', 'yr': '32'})
        
        obstacle2 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle2, 'vertex', {'xr': '0', 'yr': '35'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '0', 'yr': '36'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '63', 'yr': '35'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '63', 'yr': '36'})
    
    elif env_type == 'doorway':
        obstacles = ET.SubElement(root, 'obstacles', {'number': '2'})
        # Add doorway walls
        obstacle1 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle1, 'vertex', {'xr': '30', 'yr': '0'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '31', 'yr': '0'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '30', 'yr': '30'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '31', 'yr': '30'})
        
        obstacle2 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle2, 'vertex', {'xr': '30', 'yr': '34'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '31', 'yr': '34'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '30', 'yr': '64'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '31', 'yr': '64'})
    
    elif env_type == 'intersection':
        obstacles = ET.SubElement(root, 'obstacles', {'number': '4'})
        # Add intersection walls
        # Top-left building
        obstacle1 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle1, 'vertex', {'xr': '0', 'yr': '0'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '25', 'yr': '0'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '25', 'yr': '25'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '0', 'yr': '25'})
        
        # Top-right building
        obstacle2 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle2, 'vertex', {'xr': '39', 'yr': '0'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '64', 'yr': '0'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '64', 'yr': '25'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '39', 'yr': '25'})
        
        # Bottom-left building
        obstacle3 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle3, 'vertex', {'xr': '0', 'yr': '39'})
        ET.SubElement(obstacle3, 'vertex', {'xr': '25', 'yr': '39'})
        ET.SubElement(obstacle3, 'vertex', {'xr': '25', 'yr': '64'})
        ET.SubElement(obstacle3, 'vertex', {'xr': '0', 'yr': '64'})
        
        # Bottom-right building
        obstacle4 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle4, 'vertex', {'xr': '39', 'yr': '39'})
        ET.SubElement(obstacle4, 'vertex', {'xr': '64', 'yr': '39'})
        ET.SubElement(obstacle4, 'vertex', {'xr': '64', 'yr': '64'})
        ET.SubElement(obstacle4, 'vertex', {'xr': '39', 'yr': '64'})
    
    # Add algorithm section
    algorithm = ET.SubElement(root, 'algorithm')
    ET.SubElement(algorithm, 'searchtype').text = 'direct'
    ET.SubElement(algorithm, 'breakingties').text = '0'
    ET.SubElement(algorithm, 'allowsqueeze').text = 'false'
    ET.SubElement(algorithm, 'cutcorners').text = 'false'
    ET.SubElement(algorithm, 'hweight').text = '1'
    ET.SubElement(algorithm, 'timestep').text = '0.1'
    ET.SubElement(algorithm, 'delta').text = '0.1'
    
    # Create XML tree and save to file
    tree = ET.ElementTree(root)
    configs_dir = Path(__file__).parent / "Methods/Social-ORCA/configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_filename = configs_dir / f'config_{env_type}_{num_robots}_robots.xml'
    
    tree.write(config_filename, encoding='utf-8', xml_declaration=True)
    print(f"\nConfiguration saved to {config_filename}")
    return config_filename

def main():
    print("Welcome to the Multi-Agent Navigation Simulator")
    print("=============================================")
    print("\nAvailable Methods:")
    print("1. Social-ORCA")
    print("2. Social-IMPC-DR")
    
    while True:
        try:
            choice = int(input("\nEnter method number (1-2): "))
            if choice in [1, 2]:
                break
            print("Invalid choice! Please enter 1 or 2.")
        except ValueError:
            print("Invalid input! Please enter a number.")
    
    # Store the original directory
    original_dir = os.getcwd()
    
    try:
        if choice == 1:
            # Ask for environment type
            print("\nAvailable environments:")
            print("1. doorway")
            print("2. hallway")
            print("3. intersection")
            
            while True:
                try:
                    env_choice = int(input("\nEnter environment type (1-3): "))
                    if env_choice in [1, 2, 3]:
                        break
                    print("Invalid choice! Please enter 1, 2, or 3.")
                except ValueError:
                    print("Invalid input! Please enter a number.")
            
            env_types = {1: 'doorway', 2: 'hallway', 3: 'intersection'}
            env_type = env_types[env_choice]
            
            # Ask for number of robots
            if env_type == 'intersection':
                print("\nIntersection Configuration:")
                print("- The intersection has four square buildings in the corners with a + shaped path between them.")
                print("- The intersection opening is at (30-34, 30-34)")
                
                while True:
                    try:
                        num_robots = int(input("\nEnter number of robots (1-4): "))
                        if 0 < num_robots <= 4:
                            break
                        print("Invalid number! Please enter a number between 1 and 4.")
                    except ValueError:
                        print("Invalid input! Please enter a number.")
                
                print("\nWould you like to:")
                print("1. Use default positions (2 robots crossing scenario)")
                print("2. Enter custom positions")
                
                while True:
                    try:
                        pos_choice = int(input("\nEnter choice (1-2): "))
                        if pos_choice in [1, 2]:
                            break
                        print("Invalid choice! Please enter 1 or 2.")
                    except ValueError:
                        print("Invalid input! Please enter a number.")
                
                if pos_choice == 1:
                    if num_robots != 2:
                        print("\nNote: Default positions are designed for 2 robots.")
                        print("Proceeding with custom positions instead.")
                        pos_choice = 2
                    else:
                        print("\nUsing default intersection scenario:")
                        print("- Robot 1 (orange): Moving right to left")
                        print("- Robot 2 (green): Moving bottom to top")
                        robot_positions = [
                            {
                                'start_x': 45.0,  # Right side
                                'start_y': 32.0,  # Middle of horizontal corridor
                                'goal_x': 15.0,   # Left side
                                'goal_y': 32.0    # Middle of horizontal corridor
                            },
                            {
                                'start_x': 32.0,  # Middle of vertical corridor
                                'start_y': 15.0,  # Bottom
                                'goal_x': 32.0,   # Middle of vertical corridor
                                'goal_y': 45.0    # Top
                            }
                        ]
                
                if pos_choice == 2:
                    print("\nEnter custom positions:")
                    print("Tips for intersection navigation:")
                    print("- Use the corridors (x=30-34 or y=30-34) to cross the intersection")
                    print("- Avoid placing robots in the walls")
                    robot_positions = []
                    for i in range(num_robots):
                        print(f"\nRobot {i+1} configuration:")
                        while True:
                            try:
                                start_x = float(input(f"Enter start X position (0-63) for robot {i+1}: "))
                                start_y = float(input(f"Enter start Y position (0-63) for robot {i+1}: "))
                                if 0 <= start_x <= 63 and 0 <= start_y <= 63:
                                    break
                                print("Invalid position! Please enter values between 0 and 63.")
                            except ValueError:
                                print("Invalid input! Please enter a number.")
                        
                        while True:
                            try:
                                goal_x = float(input(f"Enter goal X position (0-63) for robot {i+1}: "))
                                goal_y = float(input(f"Enter goal Y position (0-63) for robot {i+1}: "))
                                if 0 <= goal_x <= 63 and 0 <= goal_y <= 63:
                                    break
                                print("Invalid position! Please enter values between 0 and 63.")
                            except ValueError:
                                print("Invalid input! Please enter a number.")
                        
                        robot_positions.append({
                            'start_x': start_x,
                            'start_y': start_y,
                            'goal_x': goal_x,
                            'goal_y': goal_y
                        })
                        print(f"Robot {i+1} will move from ({start_x}, {start_y}) to ({goal_x}, {goal_y})")
            else:
                while True:
                    try:
                        num_robots = int(input("\nEnter number of robots (1-4): "))
                        if 0 < num_robots <= 4:
                            break
                        print("Invalid number! Please enter a number between 1 and 4.")
                    except ValueError:
                        print("Invalid input! Please enter a number.")
                
                # Print environment-specific instructions
                if env_type == 'hallway':
                    print("\nHallway Configuration:")
                    print("- The hallway has walls at y=31-32 and y=35-36")
                    print("- Robots should stay at y=33.5 (middle of hallway)")
                    print("- X coordinates should be between 0 and 63")
                elif env_type == 'doorway':
                    print("\nDoorway Configuration:")
                    print("- The doorway has walls at x=30-31 with a gap at y=30-34")
                    print("- Y coordinates should be between 0 and 63")
                    print("- X coordinates should be between 0 and 63")
                
                # Get robot positions
                robot_positions = []
                for i in range(num_robots):
                    print(f"\nRobot {i+1} configuration:")
                    
                    # Get start position
                    while True:
                        try:
                            if env_type == 'hallway':
                                start_x = float(input(f"Enter start X position (0-63) for robot {i+1}: "))
                                start_y = 33.5  # Fixed Y position for hallway
                            else:
                                start_x = float(input(f"Enter start X position (0-63) for robot {i+1}: "))
                                start_y = float(input(f"Enter start Y position (0-63) for robot {i+1}: "))
                            
                            if 0 <= start_x <= 63 and 0 <= start_y <= 63:
                                break
                            print("Invalid position! Please enter values between 0 and 63.")
                        except ValueError:
                            print("Invalid input! Please enter a number.")
                    
                    # Get goal position
                    while True:
                        try:
                            if env_type == 'hallway':
                                goal_x = float(input(f"Enter goal X position (0-63) for robot {i+1}: "))
                                goal_y = 33.5  # Fixed Y position for hallway
                            else:
                                goal_x = float(input(f"Enter goal X position (0-63) for robot {i+1}: "))
                                goal_y = float(input(f"Enter goal Y position (0-63) for robot {i+1}: "))
                            
                            if 0 <= goal_x <= 63 and 0 <= goal_y <= 63:
                                break
                            print("Invalid position! Please enter values between 0 and 63.")
                        except ValueError:
                            print("Invalid input! Please enter a number.")
                    
                    robot_positions.append({
                        'start_x': start_x,
                        'start_y': start_y,
                        'goal_x': goal_x,
                        'goal_y': goal_y
                    })
                    print(f"Robot {i+1} will move from ({start_x}, {start_y}) to ({goal_x}, {goal_y})")
            
            print("\nGenerating configuration file...")
            config_file = generate_config(env_type, num_robots, robot_positions)
            print(f"\nConfiguration file generated: {config_file}")
            
            # Run the simulation
            run_social_orca(config_file, num_robots)
        else:
            # Ask for environment type for IMPC-DR
            print("\nAvailable environments:")
            print("1. doorway")
            print("2. hallway")
            print("3. intersection")
            
            while True:
                try:
                    env_choice = int(input("\nEnter environment type (1-3): "))
                    if env_choice in [1, 2, 3]:
                        break
                    print("Invalid choice! Please enter 1, 2, or 3.")
                except ValueError:
                    print("Invalid input! Please enter a number.")
            
            env_types = {1: 'doorway', 2: 'hallway', 3: 'intersection'}
            env_type = env_types[env_choice]
            
            run_social_impc_dr(env_type)
    finally:
        # Always return to the original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main() 