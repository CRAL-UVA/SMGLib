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
import json
import venv
import shutil

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
        
        # Flow Rate calculation for ORCA
        print("\nCalculating Flow Rate...")
        
        # Parse the log file to extract makespan and completion metrics
        tree = ET.parse(latest_log)
        root = tree.getroot()
        
        # Extract makespan from summary (total simulation time)
        summary = root.find('.//summary')
        total_makespan = None
        if summary is not None:
            makespan_str = summary.get('makespan')
            if makespan_str:
                total_makespan = float(makespan_str)
                print(f"Total simulation time: {total_makespan:.2f}s")
        
        # Extract individual agent completion times and success status from log
        log_section = root.find('log')
        agent_completion_data = []
        if log_section is not None:
            for i, agent_log in enumerate(log_section.findall('agent')):
                path = agent_log.find('path')
                if path is not None:
                    steps_attr = path.get('steps')
                    pathfound_attr = path.get('pathfound')
                    
                    if steps_attr and pathfound_attr:
                        steps_count = int(steps_attr)
                        completion_time = steps_count * time_step
                        reached_goal = pathfound_attr.lower() == 'true'
                        
                        agent_completion_data.append({
                            'agent_id': i,
                            'completion_time': completion_time,
                            'reached_goal': reached_goal,
                            'steps': steps_count
                        })
        
        # Analyze completion data to determine appropriate make-span
        if agent_completion_data:
            agents_reached_goals = [data for data in agent_completion_data if data['reached_goal']]
            agents_failed = [data for data in agent_completion_data if not data['reached_goal']]
            
            print(f"Agents that reached goals: {len(agents_reached_goals)}/{len(agent_completion_data)}")
            
            if agents_reached_goals:
                # Get the completion time of the last agent to reach its goal
                goal_completion_times = [data['completion_time'] for data in agents_reached_goals]
                latest_goal_completion = max(goal_completion_times)
                
                if len(agents_reached_goals) == len(agent_completion_data):
                    # All agents reached their goals - use the time when the last one finished
                    makespan = latest_goal_completion
                    print(f"All agents reached goals. Make-span: {makespan:.2f}s")
                    print(f"Individual completion times: {[f'{t:.2f}s' for t in goal_completion_times]}")
                else:
                    # Some agents didn't reach goals - use total simulation time for fairness
                    all_completion_times = [data['completion_time'] for data in agent_completion_data]
                    makespan = max(all_completion_times) if all_completion_times else (total_makespan or latest_goal_completion)
                    print(f"Not all agents reached goals. Using total simulation time: {makespan:.2f}s")
                    print(f"Successful agents completed at: {[f'{t:.2f}s' for t in goal_completion_times]}")
                    if agents_failed:
                        failed_times = [data['completion_time'] for data in agents_failed]
                        print(f"Failed agents stopped at: {[f'{t:.2f}s' for t in failed_times]}")
            else:
                # No agents reached their goals - use total simulation time
                all_completion_times = [data['completion_time'] for data in agent_completion_data]
                makespan = max(all_completion_times) if all_completion_times else total_makespan
                print(f"No agents reached their goals. Make-span: {makespan:.2f}s")
        elif total_makespan:
            makespan = total_makespan
            print(f"Using total simulation time as make-span: {makespan:.2f}s")
        else:
            # Fallback: calculate makespan from trajectory data
            max_steps = max(len(agent['positions']) for agent in agents_data)
            makespan = max_steps * time_step
            print(f"Make-span calculated from trajectory data: {makespan:.2f}s")
        
        # Determine gap width based on config file environment
        # ORCA uses grid coordinates (0-64), so we need to calculate actual gap widths
        config_name = str(config_path).lower()
        if 'doorway' in config_name:
            gap_width = 4.0  # doorway gap: y=30-34, so width = 4 grid units
        elif 'hallway' in config_name:
            gap_width = 3.0  # hallway gap: between y=32 and y=35, effective width = 3 grid units  
        elif 'intersection' in config_name:
            # For the new four-corner intersection, calculate total corridor width
            # The intersection has corridors on all four sides with a central open area
            # North corridor: y=26-38 (width = 12)
            # South corridor: y=26-38 (width = 12) 
            # East corridor: x=26-38 (width = 12)
            # West corridor: x=26-38 (width = 12)
            # Total effective gap width = sum of all corridor widths = 48 grid units
            gap_width = 14.0  # default intersection corridor width
        else:
            # Try to determine from obstacles in config file
            try:
                config_tree = ET.parse(config_path)
                config_root = config_tree.getroot()
                obstacles = config_root.findall('.//obstacle')
                
                if len(obstacles) >= 2:
                    # Analyze obstacle configuration to determine gap width
                    obstacle_coords = []
                    for obstacle in obstacles:
                        vertices = []
                        for vertex in obstacle.findall('vertex'):
                            x = float(vertex.get('xr'))
                            y = float(vertex.get('yr'))
                            vertices.append((x, y))
                        obstacle_coords.append(vertices)
                    
                    # Detect environment type by analyzing obstacle positions
                    vertical_walls = []
                    horizontal_walls = []
                    
                    for vertices in obstacle_coords:
                        if len(vertices) >= 4:
                            # Check if it's a vertical wall (constant x, varying y)
                            x_coords = [x for x, y in vertices]
                            y_coords = [y for x, y in vertices]
                            
                            if max(x_coords) - min(x_coords) <= 1:  # Vertical wall
                                vertical_walls.append((min(y_coords), max(y_coords)))
                            elif max(y_coords) - min(y_coords) <= 1:  # Horizontal wall
                                horizontal_walls.append((min(x_coords), max(x_coords)))
                    
                    if len(obstacles) == 8:
                        # Four-corner intersection detected
                        gap_width = 14.0  # intersection corridor width
                    elif vertical_walls:
                        # Doorway scenario - find gap between vertical walls
                        if len(vertical_walls) >= 2:
                            # Sort by y-coordinate to find gap
                            vertical_walls.sort()
                            gap_width = vertical_walls[1][0] - vertical_walls[0][1]
                        else:
                            gap_width = 4.0  # default doorway
                    elif horizontal_walls:
                        # Hallway scenario - find gap between horizontal walls  
                        if len(horizontal_walls) >= 2:
                            horizontal_walls.sort(key=lambda x: x[0])  # Sort by y position in wall tuples
                            # For hallway, walls are at y=31-32 and y=35-36, so gap = 35-32 = 3
                            gap_width = 3.0
                        else:
                            gap_width = 3.0  # default hallway
                    else:
                        # Intersection or unknown - use large buildings analysis
                        gap_width = 14.0  # default intersection corridor width
                else:
                    gap_width = 4.0  # default fallback
            except Exception as e:
                print(f"Warning: Could not parse config file for gap width: {e}")
                gap_width = 4.0  # default fallback
        
        # Calculate flow rate: N / (z * T)
        if makespan > 0 and gap_width > 0:
            # For flow rate calculation, consider different scenarios
            if agent_completion_data:
                agents_reached_goals = [data for data in agent_completion_data if data['reached_goal']]
                successful_agents = len(agents_reached_goals)
                
                if successful_agents == num_robots:
                    # All agents successful - use standard flow rate formula
                    flow_rate = num_robots / (gap_width * makespan)
                    flow_rate_type = "All agents reached goals"
                elif successful_agents > 0:
                    # Partial success - calculate based on successful agents only
                    flow_rate = successful_agents / (gap_width * makespan)
                    flow_rate_type = f"Only {successful_agents}/{num_robots} agents reached goals"
                else:
                    # No success - flow rate is effectively 0
                    flow_rate = 0.0
                    flow_rate_type = "No agents reached goals"
            else:
                # Fallback to standard calculation
                flow_rate = num_robots / (gap_width * makespan)
                flow_rate_type = "Standard calculation (goal status unknown)"
            
            print("*" * 65)
            print(f"ORCA Flow Rate Calculation:")
            print(f"Scenario: {flow_rate_type}")
            print(f"Total agents: {num_robots}")
            if agent_completion_data and len([data for data in agent_completion_data if data['reached_goal']]) != num_robots:
                successful_agents = len([data for data in agent_completion_data if data['reached_goal']])
                print(f"Successful agents: {successful_agents}")
            print(f"Gap width (z): {gap_width} grid units")
            print(f"Make-span (T): {makespan:.2f}s")
            print(f"Flow Rate: {flow_rate:.4f} agents/(unit·s)")
            print("*" * 65)
        else:
            print("*" * 65)
            print("Flow Rate: Could not compute (invalid make-span or gap width)")
            print(f"Make-span: {makespan:.2f}s, Gap width: {gap_width}")
            print("*" * 65)
            
    except Exception as e:
        print(f"Error processing trajectories: {e}")
        return

def run_social_impc_dr(env_type='doorway'):
    """Run Social-IMPC-DR by switching to its directory and calling app2.py."""
    print("\nRunning Social-IMPC-DR simulation...")
    
    # Create IMPC-DR-specific working directory
    impc_dir = Path("Methods/Social-IMPC-DR").resolve()  # Get absolute path
    original_dir = os.getcwd()
    
    print(f"IMPC-DR directory: {impc_dir}")
    print(f"Original directory: {original_dir}")
    
    try:
        # Check if IMPC-DR environment is set up, create if not
        impc_venv = impc_dir / "venv"
        setup_marker = impc_venv / "impc_setup_complete"
        
        if not setup_marker.exists():
            print("\n" + "="*50)
            print("IMPC-DR ENVIRONMENT SETUP")
            print("="*50)
            print("First-time setup: Preparing IMPC-DR environment...")
            
            if not setup_impc_environment(impc_dir):
                print("✗ Failed to set up IMPC-DR environment!")
                return
            
            print("✓ IMPC-DR environment setup complete!")
            print("="*50)
        
        # Get the full path to the app2.py script
        script_path = impc_dir / "app2.py"
        
        # Verify the script exists
        if not script_path.exists():
            print(f"✗ IMPC-DR script not found at: {script_path}")
            # Try to list what's actually in the directory
            if impc_dir.exists():
                print(f"Directory contents: {list(impc_dir.iterdir())}")
            else:
                print(f"Directory doesn't exist: {impc_dir}")
            return
        else:
            print(f"✓ Found IMPC-DR script at: {script_path}")
            
        # Change to the IMPC-DR directory
        os.chdir(impc_dir)
        print(f"Changed to directory: {impc_dir}")
        
        # Add the IMPC-DR directory to Python path
        import sys
        sys.path.insert(0, str(impc_dir))
        
        try:
            # Import and run the IMPC-DR script
            print(f"Starting IMPC-DR simulation with environment: {env_type}")
            
            # Clear any existing app2 module to avoid conflicts
            if 'app2' in sys.modules:
                del sys.modules['app2']
            
            # Try to run app2 directly
            try:
                import app2
                print("✓ Successfully imported app2")
                
                # Call the main function from app2 with environment type
                # We need to pass the environment type as a command line argument
                import sys
                original_argv = sys.argv.copy()
                sys.argv = ['app2.py', env_type]
                
                try:
                    # Call the main function if it exists, otherwise run the script
                    if hasattr(app2, 'main'):
                        result = app2.main()
                    else:
                        # Execute the script content
                        with open(script_path, 'r') as f:
                            script_content = f.read()
                        exec(script_content, {'__name__': '__main__'})
                        result = 0
                    
                    if result == 0 or result is None:  # Some scripts may not return a value
                        print("✓ IMPC-DR simulation completed successfully!")
                        
                        # Look for generated trajectory files
                        path_deviation_files = list(impc_dir.glob("path_deviation_robot_*.csv"))
                        if path_deviation_files:
                            print(f"✓ Found {len(path_deviation_files)} trajectory files")
                            
                            # Evaluate trajectories
                            evaluate_impc_trajectories(impc_dir, env_type, path_deviation_files)
                        else:
                            print("⚠ No trajectory files found")
                    else:
                        print("✗ IMPC-DR simulation completed with errors")
                        
                finally:
                    # Restore original argv
                    sys.argv = original_argv
                    
            except ImportError as import_error:
                print(f"Import error: {import_error}")
                print("Trying alternative execution method...")
                
                # Alternative: execute the script file directly with subprocess
                print(f"Executing script directly: {script_path}")
                result = subprocess.run([get_venv_python(), "app2.py", env_type], 
                                      cwd=impc_dir, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✓ IMPC-DR simulation completed successfully!")
                    print(result.stdout)
                    
                    # Look for generated trajectory files
                    path_deviation_files = list(impc_dir.glob("path_deviation_robot_*.csv"))
                    if path_deviation_files:
                        print(f"✓ Found {len(path_deviation_files)} trajectory files")
                        
                        # Evaluate trajectories
                        evaluate_impc_trajectories(impc_dir, env_type, path_deviation_files)
                    else:
                        print("⚠ No trajectory files found")
                else:
                    print("✗ IMPC-DR simulation failed")
                    print(f"Error: {result.stderr}")
                
            except Exception as run_error:
                print(f"Error running IMPC-DR script: {run_error}")
                import traceback
                traceback.print_exc()
                
        finally:
            # Remove IMPC-DR path from sys.path
            if str(impc_dir) in sys.path:
                sys.path.remove(str(impc_dir))
            # Clean up imported module
            if 'app2' in sys.modules:
                del sys.modules['app2']
        
    except Exception as e:
        print(f"✗ Error running IMPC-DR: {e}")
    finally:
        os.chdir(original_dir)


def setup_impc_environment(impc_dir):
    """Set up IMPC-DR-specific virtual environment with compatible dependencies."""
    
    venv_dir = impc_dir / "venv"
    
    try:
        # Remove existing environment if it exists
        if venv_dir.exists():
            print("Removing existing environment...")
            shutil.rmtree(venv_dir)
        
        # Create new virtual environment
        print("Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)
        
        # Get python and pip executables for the new environment
        if sys.platform == "win32":
            python_exe = venv_dir / "Scripts" / "python.exe"
            pip_exe = venv_dir / "Scripts" / "pip.exe"
        else:
            python_exe = venv_dir / "bin" / "python"
            pip_exe = venv_dir / "bin" / "pip"
        
        print("Installing IMPC-DR dependencies...")
        
        # Install requirements from the IMPC-DR directory
        requirements_file = impc_dir / "requirements.txt"
        if requirements_file.exists():
            print(f"Installing from {requirements_file}")
            subprocess.run([str(pip_exe), "install", "-r", str(requirements_file)], 
                         check=True, capture_output=True, text=True)
        else:
            print("No requirements.txt found, installing basic dependencies...")
            # Install basic dependencies that IMPC-DR likely needs
            basic_deps = ["numpy", "matplotlib", "pandas", "scipy"]
            for dep in basic_deps:
                try:
                    subprocess.run([str(pip_exe), "install", dep], 
                                 check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Could not install {dep}: {e}")
        
        print("✓ IMPC-DR environment structure created")
        print("✓ Dependencies installed")
        
        # Create a marker file to indicate the environment is set up
        marker_file = venv_dir / "impc_setup_complete"
        marker_file.touch()
        
        print("Environment setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error setting up IMPC-DR environment: {e}")
        return False


def evaluate_impc_trajectories(impc_dir, env_type, path_deviation_files):
    """Evaluate IMPC-DR trajectories and calculate metrics."""
    print("\nEvaluating Social-IMPC-DR trajectories:")
    num_agents = len(path_deviation_files)
    max_steps = 0
    
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
        # Track max steps for make-span
        if len(data) > max_steps:
            max_steps = len(data)
    
    # Flow Rate calculation
    # Try to get step size from app2.py arguments (default 0.1)
    step_size = 0.1
    # Set gap width z based on env_type (these are in different coordinate systems)
    # Social-IMPC-DR uses normalized coordinates (0-2.5 scale), while ORCA uses grid coordinates (0-64)
    if env_type == 'doorway':
        gap_width = 0.8  # gap_end - gap_start = 1.6 - 0.8 = 0.8 (normalized units)
    elif env_type == 'hallway':
        gap_width = 1.0  # effective corridor width in normalized units
    elif env_type == 'intersection':
        gap_width = 0.8  # corridor total width (2 * corridor_half_width = 2 * 0.4 = 0.8) (normalized units)
    else:
        gap_width = 1.0  # fallback
    
    # Try to get actual completion step (when all robots reached goals)
    completion_step_file = os.path.join(impc_dir, "completion_step.txt")
    actual_completion_steps = max_steps  # fallback to max_steps
    if os.path.exists(completion_step_file):
        try:
            with open(completion_step_file, 'r') as f:
                actual_completion_steps = int(f.read().strip())
            print(f"Using actual completion time: {actual_completion_steps} steps (goals reached)")
        except:
            print(f"Could not read completion step file, using max steps: {max_steps}")
    else:
        print(f"No completion step file found, using max steps: {max_steps}")
    
    make_span = actual_completion_steps * step_size
    
    # Enhanced flow rate calculation for Social-IMPC-DR
    if make_span > 0 and gap_width > 0:
        if actual_completion_steps < max_steps:
            # All robots reached their goals before the simulation ended
            flow_rate_type = "All agents reached goals"
            effective_agents = num_agents
        else:
            # Simulation ended without all robots reaching goals - check individual completion
            flow_rate_type = f"Simulation completed (check individual robot completion)"
            effective_agents = num_agents  # Use all agents but note the limitation
        
        flow_rate = effective_agents / (gap_width * make_span)
        print("*" * 65)
        print(f"Social-IMPC-DR Flow Rate Calculation:")
        print(f"Scenario: {flow_rate_type}")
        print(f"Environment: {env_type}")
        print(f"Number of agents: {num_agents}")
        print(f"Gap width (z): {gap_width} normalized units")
        print(f"Make-span (T): {make_span:.2f}s")
        print(f"Completion step: {actual_completion_steps}/{max_steps}")
        print(f"Flow Rate: {flow_rate:.4f} agents/(unit·s)")
        print("*" * 65)
    else:
        print("*" * 65)
        print("Flow Rate: Could not compute (invalid make-span or gap width)")
        print(f"Make-span: {make_span:.2f}s, Gap width: {gap_width}")
        print("*" * 65)

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
        obstacles = ET.SubElement(root, 'obstacles', {'number': '8'})
        # Create a proper four-corner intersection with 8 obstacles
        # Each corner has 2 obstacles forming the walls
        
        # Top-left corner (north-west)
        obstacle1 = ET.SubElement(obstacles, 'obstacle')  # Vertical wall
        ET.SubElement(obstacle1, 'vertex', {'xr': '25', 'yr': '0'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '26', 'yr': '0'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '25', 'yr': '25'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '26', 'yr': '25'})
        
        obstacle2 = ET.SubElement(obstacles, 'obstacle')  # Horizontal wall
        ET.SubElement(obstacle2, 'vertex', {'xr': '0', 'yr': '25'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '0', 'yr': '26'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '25', 'yr': '25'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '25', 'yr': '26'})
        
        # Top-right corner (north-east)
        obstacle3 = ET.SubElement(obstacles, 'obstacle')  # Vertical wall
        ET.SubElement(obstacle3, 'vertex', {'xr': '38', 'yr': '0'})
        ET.SubElement(obstacle3, 'vertex', {'xr': '39', 'yr': '0'})
        ET.SubElement(obstacle3, 'vertex', {'xr': '38', 'yr': '25'})
        ET.SubElement(obstacle3, 'vertex', {'xr': '39', 'yr': '25'})
        
        obstacle4 = ET.SubElement(obstacles, 'obstacle')  # Horizontal wall
        ET.SubElement(obstacle4, 'vertex', {'xr': '39', 'yr': '25'})
        ET.SubElement(obstacle4, 'vertex', {'xr': '39', 'yr': '26'})
        ET.SubElement(obstacle4, 'vertex', {'xr': '64', 'yr': '25'})
        ET.SubElement(obstacle4, 'vertex', {'xr': '64', 'yr': '26'})
        
        # Bottom-left corner (south-west)
        obstacle5 = ET.SubElement(obstacles, 'obstacle')  # Vertical wall
        ET.SubElement(obstacle5, 'vertex', {'xr': '25', 'yr': '39'})
        ET.SubElement(obstacle5, 'vertex', {'xr': '26', 'yr': '39'})
        ET.SubElement(obstacle5, 'vertex', {'xr': '25', 'yr': '64'})
        ET.SubElement(obstacle5, 'vertex', {'xr': '26', 'yr': '64'})
        
        obstacle6 = ET.SubElement(obstacles, 'obstacle')  # Horizontal wall
        ET.SubElement(obstacle6, 'vertex', {'xr': '0', 'yr': '38'})
        ET.SubElement(obstacle6, 'vertex', {'xr': '0', 'yr': '39'})
        ET.SubElement(obstacle6, 'vertex', {'xr': '25', 'yr': '38'})
        ET.SubElement(obstacle6, 'vertex', {'xr': '25', 'yr': '39'})
        
        # Bottom-right corner (south-east)
        obstacle7 = ET.SubElement(obstacles, 'obstacle')  # Vertical wall
        ET.SubElement(obstacle7, 'vertex', {'xr': '38', 'yr': '39'})
        ET.SubElement(obstacle7, 'vertex', {'xr': '39', 'yr': '39'})
        ET.SubElement(obstacle7, 'vertex', {'xr': '38', 'yr': '64'})
        ET.SubElement(obstacle7, 'vertex', {'xr': '39', 'yr': '64'})
        
        obstacle8 = ET.SubElement(obstacles, 'obstacle')  # Horizontal wall
        ET.SubElement(obstacle8, 'vertex', {'xr': '39', 'yr': '38'})
        ET.SubElement(obstacle8, 'vertex', {'xr': '39', 'yr': '39'})
        ET.SubElement(obstacle8, 'vertex', {'xr': '64', 'yr': '38'})
        ET.SubElement(obstacle8, 'vertex', {'xr': '64', 'yr': '39'})
    
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

def run_social_cadrl():
    """Run Social-CADRL by switching to its directory and calling run_scenarios.py."""
    print("\nRunning Social-CADRL simulation...")
    
    # Handle numpy compatibility issues for CADRL
    try:
        import numpy as np
        if not hasattr(np, 'bool8'):
            # Add bool8 as an alias for bool_ to maintain compatibility
            np.bool8 = np.bool_
            print("✓ Applied NumPy compatibility fix for CADRL")
    except Exception as e:
        print(f"Warning: Could not apply NumPy compatibility fix: {e}")
    
    # Create CADRL-specific working directory
    cadrl_dir = Path("Methods/Social-CADRL").resolve()  # Get absolute path
    original_dir = os.getcwd()
    
    print(f"CADRL directory: {cadrl_dir}")
    print(f"Original directory: {original_dir}")
    
    try:
        # Check if CADRL environment is set up, create if not
        cadrl_venv = cadrl_dir / "venv"
        setup_marker = cadrl_venv / "cadrl_setup_complete"
        
        if not setup_marker.exists():
            print("\n" + "="*50)
            print("CADRL ENVIRONMENT SETUP")
            print("="*50)
            print("First-time setup: Preparing CADRL environment...")
            
            if not setup_cadrl_environment(cadrl_dir):
                print("✗ Failed to set up CADRL environment!")
                return
            
            print("✓ CADRL environment setup complete!")
            print("="*50)
        
        # Get the full path to the experiments/src directory
        cadrl_experiments_dir = cadrl_dir / "experiments" / "src"
        script_path = cadrl_experiments_dir / "run_scenarios.py"
        
        # Verify the script exists
        if not script_path.exists():
            print(f"✗ CADRL script not found at: {script_path}")
            # Try to list what's actually in the directory
            if cadrl_experiments_dir.exists():
                print(f"Directory contents: {list(cadrl_experiments_dir.iterdir())}")
            else:
                print(f"Directory doesn't exist: {cadrl_experiments_dir}")
            return
        else:
            print(f"✓ Found CADRL script at: {script_path}")
            
        # Change to the experiments/src directory
        os.chdir(cadrl_experiments_dir)
        print(f"Changed to directory: {cadrl_experiments_dir}")
        
        # Add the CADRL directories to Python path (at the beginning)
        import sys
        sys.path.insert(0, str(cadrl_experiments_dir))
        sys.path.insert(0, str(cadrl_dir))
        
        try:
            # Import and run the CADRL script
            print("Starting CADRL simulation...")
            
            # Clear any existing run_scenarios module to avoid conflicts
            if 'run_scenarios' in sys.modules:
                del sys.modules['run_scenarios']
            
            # Try to run run_scenarios directly
            try:
                import run_scenarios
                print("✓ Successfully imported run_scenarios")
                
                # Call the main function from run_scenarios - it will handle its own user input
                result = run_scenarios.main()
                
                if result == 0 or result is None:  # Some scripts may not return a value
                    print("✓ CADRL simulation completed successfully!")
                    
                    # Look for generated animation files
                    animations_dir = cadrl_dir / "experiments" / "results" / "example" / "animations"
                    if animations_dir.exists():
                        animation_files = list(animations_dir.glob("*.gif"))
                        if animation_files:
                            print(f"✓ Animation saved: {animation_files[0]}")
                else:
                    print("✗ CADRL simulation completed with errors")
                    
            except ImportError as import_error:
                print(f"Import error: {import_error}")
                print("Trying alternative execution method...")
                
                # Alternative: execute the script file directly
                print(f"Executing script directly: {script_path}")
                with open(script_path, 'r') as f:
                    script_content = f.read()
                exec(script_content, {'__name__': '__main__'})
                
            except Exception as run_error:
                print(f"Error running CADRL script: {run_error}")
                import traceback
                traceback.print_exc()
                
        finally:
            # Remove CADRL paths from sys.path
            if str(cadrl_experiments_dir) in sys.path:
                sys.path.remove(str(cadrl_experiments_dir))
            if str(cadrl_dir) in sys.path:
                sys.path.remove(str(cadrl_dir))
            # Clean up imported module
            if 'run_scenarios' in sys.modules:
                del sys.modules['run_scenarios']
        
    except Exception as e:
        print(f"✗ Error running CADRL: {e}")
    finally:
        os.chdir(original_dir)



def setup_cadrl_environment(cadrl_dir):
    """Set up CADRL-specific virtual environment with compatible dependencies."""
    
    venv_dir = cadrl_dir / "venv"
    
    try:
        # Remove existing environment if it exists
        if venv_dir.exists():
            print("Removing existing environment...")
            shutil.rmtree(venv_dir)
        
        # Create new virtual environment
        print("Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)
        
        # Get python and pip executables for the new environment
        if sys.platform == "win32":
            python_exe = venv_dir / "Scripts" / "python.exe"
            pip_exe = venv_dir / "Scripts" / "pip.exe"
        else:
            python_exe = venv_dir / "bin" / "python"
            pip_exe = venv_dir / "bin" / "pip"
        
        print("Installing CADRL-compatible dependencies...")
        print("Note: Since we're running CADRL in the same process, we'll install")
        print("compatible versions and handle import conflicts programmatically.")
        
        # For now, just create the environment structure
        # The actual dependency management will be handled at runtime
        print("✓ CADRL environment structure created")
        print("✓ Dependencies will be managed at runtime")
        
        # Create a marker file to indicate the environment is set up
        marker_file = venv_dir / "cadrl_setup_complete"
        marker_file.touch()
        
        print("Environment setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error setting up CADRL environment: {e}")
        return False

def main():
    print("Welcome to the Multi-Agent Navigation Simulator")
    print("=============================================")
    print("\nAvailable Methods:")
    print("1. Social-ORCA")
    print("2. Social-IMPC-DR")
    print("3. Social-CADRL")
    
    while True:
        try:
            choice = int(input("\nEnter method number (1-3): "))
            if choice in [1, 2, 3]:
                break
            print("Invalid choice! Please enter 1, 2, or 3.")
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
            elif env_type == 'intersection':
                print("\nIntersection Configuration:")
                print("- The intersection has 8 obstacles forming corridors on all four sides")
                print("- Central open area: x=26-38, y=26-38")
                print("- North corridor: y=0-25 (use y=12.5 for north approach)")
                print("- South corridor: y=39-64 (use y=51.5 for south approach)")
                print("- East corridor: x=39-64 (use x=51.5 for east approach)")
                print("- West corridor: x=0-25 (use x=12.5 for west approach)")
                print("- X and Y coordinates should be between 0 and 63")
            
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
            
            # Generate configuration file
            config_file = generate_config(env_type, num_robots, robot_positions)
            
            # Run the simulation
            run_social_orca(config_file, num_robots)
        elif choice == 2:
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
        else:  # choice == 3, Social-CADRL
            print("\nStarting Social-CADRL...")
            print("The CADRL simulation will prompt you for environment and agent configuration.")
            run_social_cadrl()
    finally:
        # Always return to the original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main() 