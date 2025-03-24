import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
import os

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

    # Plotting (only if needed)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Path Deviation and Average Delta Velocity")
    parser.add_argument("--path_deviation_csv", type=str, help="CSV file for path deviation")
    parser.add_argument("--avg_delta_velocity_csv", type=str, help="CSV file for average delta velocity")
    parser.add_argument("--evaluate_path_deviation", action="store_true", help="Evaluate path deviation")
    parser.add_argument("--evaluate_avg_delta_velocity", action="store_true", help="Evaluate average delta velocity")
    parser.add_argument("--save_path_deviation_plot", action="store_true", help="Save path deviation plot")
   parser.add_argument("--method", type=str, help="Method name")
   parser.add_argument("--scenario", type=str, help="Scenario name")
   args = parser.parse_args()

    if args.evaluate_path_deviation and args.save_path_deviation_plot:
        plt, _, _ = calculate_path_deviation(args.path_deviation_csv)
        if args.save_path_deviation_plot:
            if not os.path.exists('path_deviation'):
                os.makedirs('path_deviation')
            plt.savefig('path_deviation/path_deviation_plot.png')
        else:
            plt.show()

    if args.evaluate_avg_delta_velocity and args.avg_delta_velocity_csv:
        calculate_avg_delta_velocity(args.avg_delta_velocity_csv)

if __name__ == "__main__":
    main()
