import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
import os
from pathlib import Path

def calculate_path_deviation(path_deviation_csv):
    data = pd.read_csv(path_deviation_csv)

    # Extract coordinates
    actual_x, actual_y = data.iloc[:, 0], data.iloc[:, 1]
    nominal_x, nominal_y = data.iloc[:, 2], data.iloc[:, 3]

    # Compute trajectory difference and L2 norm
    diff_x, diff_y = actual_x - nominal_x, actual_y - nominal_y
    l2_norm = np.sqrt(diff_x**2 + diff_y**2)

    # Calculate Hausdorff distance
    actual_trajectory = np.column_stack((actual_x, actual_y))
    nominal_trajectory = np.column_stack((nominal_x, nominal_y))
    hausdorff_dist = directed_hausdorff(actual_trajectory, nominal_trajectory)[0]

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-darkgrid')
    plt.plot(actual_x, actual_y, linestyle='-', linewidth=2, color='blue', label='Actual Trajectory')
    plt.plot(nominal_x, nominal_y, linestyle='-', linewidth=2, color='green', label='Nominal Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory Difference')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(min(min(actual_x), min(nominal_x)) - 1, max(max(actual_x), max(nominal_x)) + 1)
    plt.ylim(min(min(actual_y), min(nominal_y)) - 1, max(max(actual_y), max(nominal_y)) + 1)
    text = 'L2 norm: {:.4f}\nHausdorff distance: {:.4f}'.format(l2_norm.sum(), hausdorff_dist)
    plt.annotate(text, xy=(0.05, 0.9), xycoords='axes fraction')

    # Print L2 Norm and Hausdorff distance
    print("*" * 65)
    print("Path Deviation")
    print(f"L2 Norm: {l2_norm.sum():.4f}")
    print(f"Hausdorff distance: {hausdorff_dist:.4f}")
    print("*" * 65)

    return plt, l2_norm.sum(), hausdorff_dist

def calculate_avg_delta_velocity(avg_delta_velocity_csv):
    data = pd.read_csv(avg_delta_velocity_csv)
    data['resultant_velocity'] = np.sqrt(data['vx']**2 + data['vy']**2)
    diffs = np.diff(data['resultant_velocity'])
    abs_diffs = np.abs(diffs)
    sum_abs_diffs = np.sum(abs_diffs)

    # Print the average delta velocity
    print("*" * 65)
    print(f"Avg delta velocity: {sum_abs_diffs:.4f}")
    print("*" * 65)

    return sum_abs_diffs

def display_gif(method, scenario):
    # Construct the path to the gif file
    gif_path = os.path.join('vis', method, scenario + '.gif')

    # Check if the gif file exists
    if os.path.exists(gif_path):
        # Display the gif using PIL
        with Image.open(gif_path) as img:
            img.show()
    else:
        print(f"Error: {gif_path} does not exist")

def evaluate_trajectories(method, scenario):
    """Evaluate trajectories for a given method and scenario."""
    if method == "Social-ORCA":
        # Look for trajectory files in the Social-ORCA logs directory
        orca_dir = Path("Methods/Social-ORCA/logs/trajectories")
        print(f"\nLooking for trajectory files in: {orca_dir}")
        if not orca_dir.exists():
            print(f"Error: Directory {orca_dir} not found")
            return
        
        # Get all robot trajectory files
        trajectory_files = list(orca_dir.glob("robot_*_trajectory.csv"))
        print(f"Found trajectory files: {[str(f) for f in trajectory_files]}")
        if not trajectory_files:
            print(f"Error: No trajectory files found in {orca_dir}")
            return
        
        print(f"\nEvaluating {len(trajectory_files)} robot trajectories...")
        
        # Create a figure with subplots for each robot
        num_robots = len(trajectory_files)
        fig, axes = plt.subplots(num_robots, 1, figsize=(12, 6*num_robots))
        if num_robots == 1:
            axes = [axes]
        
        total_l2_norm = 0
        total_hausdorff = 0
        
        for i, traj_file in enumerate(trajectory_files):
            print(f"\nProcessing trajectory file: {traj_file}")
            try:
                _, l2_norm, hausdorff_dist = calculate_path_deviation(traj_file)
                total_l2_norm += l2_norm
                total_hausdorff += hausdorff_dist
                
                # Get the data for plotting
                data = pd.read_csv(traj_file)
                actual_x, actual_y = data.iloc[:, 0], data.iloc[:, 1]
                nominal_x, nominal_y = data.iloc[:, 2], data.iloc[:, 3]
                
                # Plot in the appropriate subplot
                axes[i].plot(actual_x, actual_y, linestyle='-', linewidth=2, color='blue', label='Actual Trajectory')
                axes[i].plot(nominal_x, nominal_y, linestyle='-', linewidth=2, color='green', label='Nominal Trajectory')
                axes[i].set_xlabel('X')
                axes[i].set_ylabel('Y')
                axes[i].set_title(f'Robot {i} Trajectory')
                axes[i].legend()
                axes[i].grid(True, linestyle='--', alpha=0.7)
                axes[i].set_xlim(min(min(actual_x), min(nominal_x)) - 1, max(max(actual_x), max(nominal_x)) + 1)
                axes[i].set_ylim(min(min(actual_y), min(nominal_y)) - 1, max(max(actual_y), max(nominal_y)) + 1)
                text = f'L2 norm: {l2_norm:.4f}\nHausdorff distance: {hausdorff_dist:.4f}'
                axes[i].annotate(text, xy=(0.05, 0.9), xycoords='axes fraction')
            except Exception as e:
                print(f"Error processing file {traj_file}: {str(e)}")
                continue
        
        plt.tight_layout()
        output_file = f'evaluation_{method}_{scenario}.png'
        plt.savefig(output_file)
        print(f"\nSaved evaluation plot to: {output_file}")
        plt.close()
        
        # Print average metrics
        print("\nAverage Metrics:")
        print(f"Average L2 Norm: {total_l2_norm/num_robots:.4f}")
        print(f"Average Hausdorff Distance: {total_hausdorff/num_robots:.4f}")
    
    elif method == "Social-IMPC-DR":
        # Look for trajectory files in the Social-IMPC-DR directory
        impc_dir = Path("Methods/Social-IMPC-DR")
        if not impc_dir.exists():
            print(f"Error: Directory {impc_dir} not found")
            return
        
        # Get the trajectory file
        traj_file = impc_dir / "path_deviation.csv"
        if not traj_file.exists():
            print(f"Error: No trajectory file found at {traj_file}")
            return
        
        print("\nEvaluating Social-IMPC-DR trajectory...")
        plt, l2_norm, hausdorff_dist = calculate_path_deviation(traj_file)
        plt.savefig(f'evaluation_{method}_{scenario}.png')
        plt.close()
        
        print("\nMetrics:")
        print(f"L2 Norm: {l2_norm:.4f}")
        print(f"Hausdorff Distance: {hausdorff_dist:.4f}")
    
    else:
        print(f"Error: Unknown method {method}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate multi-agent navigation methods')
    parser.add_argument('--method', choices=['Social-ORCA', 'Social-IMPC-DR'], required=True,
                      help='Method to evaluate')
    parser.add_argument('--scenario', required=True,
                      help='Scenario name (e.g., doorway_2_robots)')
    
    args = parser.parse_args()
    
    evaluate_trajectories(args.method, args.scenario)

if __name__ == "__main__":
    main()