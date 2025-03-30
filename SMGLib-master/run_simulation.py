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
            for step in path.findall('step'):
                pos = [float(step.get('xr')), float(step.get('yr'))]
                next_pos = [float(step.get('next.xr')), float(step.get('next.yr'))]
                # Calculate velocity as difference between current and next position
                vel = [next_pos[0] - pos[0], next_pos[1] - pos[1]]
                positions.append(pos)
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
    
    # Generate CSV for each agent
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
        print(f"Generated trajectory CSV for robot {agent['id']}: {output_csv}")

def get_num_robots_from_config(config_file):
    """Extract number of robots from config file."""
    tree = ET.parse(config_file)
    root = tree.getroot()
    agents = root.findall('.//agent')
    return len(agents)

def run_social_orca(config_file, num_robots):
    print("\nRunning Social-ORCA Simulation")
    print("=============================")
    
    # Change to Social-ORCA directory
    orca_dir = Path("Methods/Social-ORCA")
    os.chdir(orca_dir)
    
    # Run the configuration generator
    subprocess.run([get_venv_python(), "generate_config.py"])
    
    # Get the most recently created config file
    config_files = sorted(Path("configs").glob("config_*.xml"), key=os.path.getctime)
    if not config_files:
        print("No configuration files found!")
        return
    
    latest_config = config_files[-1]
    print(f"\nUsing configuration file: {latest_config}")
    
    try:
        num_robots = get_num_robots_from_config(latest_config)
    except Exception as e:
        print(f"Error reading config file: {e}")
        return
    
    # Run the simulation
    print(f"\nRunning simulation with {num_robots} robots...")
    cmd = f"cd {orca_dir} && ./build/SocialORCA {latest_config} {num_robots}"
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
        generate_orca_csvs(latest_log, output_dir)
        print(f"\nTrajectory CSV files generated in: {output_dir}")
        
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
    except Exception as e:
        print(f"Error generating CSV files: {e}")

def run_social_impc_dr():
    print("\nRunning Social-IMPC-DR Simulation")
    print("=================================")
    
    # Change to Social-IMPC-DR directory using absolute path
    impc_dir = Path(__file__).parent / "Methods" / "Social-IMPC-DR"
    if not impc_dir.exists():
        print(f"Error: Directory {impc_dir} not found!")
        return
    
    print(f"Changing to directory: {impc_dir}")
    os.chdir(impc_dir)
    
    # Run app2.py directly to allow for user input
    subprocess.run([get_venv_python(), "app2.py"])
    
    # Evaluate trajectory if available
    path_deviation_file = impc_dir / "path_deviation.csv"
    if path_deviation_file.exists():
        print("\nEvaluating Social-IMPC-DR trajectory:")
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
        print("Social-IMPC-DR Path Deviation Metrics:")
        print(f"L2 Norm: {l2_norm:.4f}")
        print(f"Hausdorff distance: {hausdorff_dist:.4f}")
        print("*" * 65)
    else:
        print(f"\nWarning: No path_deviation.csv file found in {impc_dir}")

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
            config_file = next(Path("Methods/Social-ORCA/configs").glob("config_*.xml"))
            num_robots = get_num_robots_from_config(config_file)
            run_social_orca(config_file, num_robots)
        else:
            run_social_impc_dr()
    finally:
        # Always return to the original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main() 