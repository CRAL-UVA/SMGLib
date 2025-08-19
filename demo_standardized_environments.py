#!/usr/bin/env python3
"""
Demonstration script for standardized environment configuration.
This script shows how the centralized environment configuration works
across different social navigation methods.
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / 'src'))
from utils import StandardizedEnvironment

def demo_environment_layouts():
    """Demonstrate the standardized environment layouts."""
    print("="*60)
    print("STANDARDIZED ENVIRONMENT CONFIGURATION DEMO")
    print("="*60)
    
    env_types = ['doorway', 'hallway', 'intersection']
    
    # Create a comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, env_type in enumerate(env_types):
        ax = axes[i]
        
        print(f"\n{env_type.upper()} Environment:")
        print(f"  Grid: X=[{StandardizedEnvironment.GRID_X_MIN}, {StandardizedEnvironment.GRID_X_MAX}], "
              f"Y=[{StandardizedEnvironment.GRID_Y_MIN}, {StandardizedEnvironment.GRID_Y_MAX}]")
        
        # Create standardized plot
        ax.set_xlim(StandardizedEnvironment.GRID_X_MIN, StandardizedEnvironment.GRID_X_MAX)
        ax.set_ylim(StandardizedEnvironment.GRID_Y_MIN, StandardizedEnvironment.GRID_Y_MAX)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'{env_type.capitalize()} Environment')
        
        # Add obstacles
        if env_type == 'doorway':
            obstacles = StandardizedEnvironment.get_doorway_obstacles()
            print(f"  Doorway: Vertical wall at x=0 with gap from y=-2 to y=2")
            print(f"  Number of obstacles: {len(obstacles)}")
        elif env_type == 'hallway':
            obstacles = StandardizedEnvironment.get_hallway_obstacles()
            print(f"  Hallway: Horizontal walls at y=-2 and y=2")
            print(f"  Number of obstacles: {len(obstacles)}")
        elif env_type == 'intersection':
            obstacles = StandardizedEnvironment.get_intersection_obstacles()
            print(f"  Intersection: + shaped corridors centered at (0,0)")
            print(f"  Number of obstacles: {len(obstacles)}")
        
        # Plot obstacles
        for obs in obstacles:
            circle = plt.Circle(obs, radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, 
                              facecolor='gray', edgecolor='black', alpha=0.7)
            ax.add_patch(circle)
        
        # Add example agent positions
        positions = StandardizedEnvironment.get_standard_agent_positions(env_type, 2)
        for j, pos in enumerate(positions):
            color = StandardizedEnvironment.AGENT_COLORS[j % len(StandardizedEnvironment.AGENT_COLORS)]
            
            # Start position
            ax.scatter(pos['start'][0], pos['start'][1], marker='s', s=100, 
                      edgecolor='black', facecolor=color, zorder=3)
            
            # Goal position
            ax.scatter(pos['goal'][0], pos['goal'][1], marker='*', s=150, 
                      edgecolor='black', facecolor=color, zorder=4)
            
            # Draw line from start to goal
            ax.plot([pos['start'][0], pos['goal'][0]], [pos['start'][1], pos['goal'][1]], 
                   color=color, linestyle='--', alpha=0.5)
            
            print(f"  Agent {j+1}: Start=({pos['start'][0]:.1f}, {pos['start'][1]:.1f}), "
                  f"Goal=({pos['goal'][0]:.1f}, {pos['goal'][1]:.1f})")
    
    plt.tight_layout()
    plt.savefig('standardized_environments_demo.png', bbox_inches='tight', dpi=300)
    plt.savefig('standardized_environments_demo.svg', bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"\nDemo plot saved as 'standardized_environments_demo.png' and 'standardized_environments_demo.svg'")

def demo_agent_parameters():
    """Demonstrate the standardized agent parameters."""
    print("\n" + "="*60)
    print("STANDARDIZED AGENT PARAMETERS")
    print("="*60)
    
    print(f"Default Agent Radius: {StandardizedEnvironment.DEFAULT_AGENT_RADIUS} m")
    print(f"Default Preferred Speed: {StandardizedEnvironment.DEFAULT_PREF_SPEED} m/s")
    print(f"Default Collision Distance: {StandardizedEnvironment.DEFAULT_COLLISION_DISTANCE} m")
    print(f"Figure Size: {StandardizedEnvironment.FIG_SIZE}")
    print(f"Animation FPS: {StandardizedEnvironment.ANIMATION_FPS}")
    print(f"Animation Interval: {StandardizedEnvironment.ANIMATION_INTERVAL} ms")
    
    print("\nAgent Colors:")
    for i, color in enumerate(StandardizedEnvironment.AGENT_COLORS):
        print(f"  Agent {i+1}: RGB{tuple(color)}")

def demo_visualization_consistency():
    """Demonstrate visualization consistency across methods."""
    print("\n" + "="*60)
    print("VISUALIZATION CONSISTENCY")
    print("="*60)
    
    # Create a sample trajectory for demonstration
    env_type = 'doorway'
    fig, ax = StandardizedEnvironment.create_standard_plot(env_type, show_obstacles=True)
    
    # Simulate agent trajectories
    t = np.linspace(0, 10, 50)
    agent1_positions = []
    agent2_positions = []
    
    for time in t:
        # Agent 1: moving from left to right through doorway
        x1 = -3 + time * 0.6  # Linear motion
        y1 = 0.5 * np.sin(time * 0.5)  # Slight oscillation
        agent1_positions.append([x1, y1])
        
        # Agent 2: moving from right to left through doorway
        x2 = 3 - time * 0.6  # Linear motion
        y2 = -0.3 * np.sin(time * 0.3)  # Slight oscillation
        agent2_positions.append([x2, y2])
    
    # Plot trajectories using standardized methods
    StandardizedEnvironment.plot_agent_trajectory(ax, agent1_positions, 0, show_trajectory=True, show_timestamps=True)
    StandardizedEnvironment.plot_agent_trajectory(ax, agent2_positions, 1, show_trajectory=True, show_timestamps=True)
    
    # Plot goals
    StandardizedEnvironment.plot_goal(ax, [3.0, 0.0], 0)
    StandardizedEnvironment.plot_goal(ax, [-3.0, 0.0], 1)
    
    # Create standardized legend
    legend_handles, legend_labels = StandardizedEnvironment.create_standard_legend(2)
    ax.legend(legend_handles, legend_labels,
              loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12, borderaxespad=0., markerscale=1.2)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.savefig('visualization_consistency_demo.png', bbox_inches='tight', dpi=300)
    plt.savefig('visualization_consistency_demo.svg', bbox_inches='tight', dpi=300)
    plt.show()
    
    print("Visualization consistency demo saved as 'visualization_consistency_demo.png' and 'visualization_consistency_demo.svg'")

def demo_method_integration():
    """Demonstrate how the standardized configuration integrates with different methods."""
    print("\n" + "="*60)
    print("METHOD INTEGRATION")
    print("="*60)
    
    print("The standardized environment configuration can be used with:")
    print("  1. Social-CADRL: Use test_cases_standardized.py")
    print("  2. Social-IMPC-DR: Use app2_standardized.py")
    print("  3. Social-ORCA: Use generate_config_standardized.py (to be created)")
    
    print("\nKey benefits:")
    print("  ✓ Consistent grid sizes across all methods")
    print("  ✓ Standardized obstacle layouts")
    print("  ✓ Unified agent parameters")
    print("  ✓ Consistent visualization styling")
    print("  ✓ Easy comparison between methods")
    print("  ✓ Centralized configuration management")

def main():
    """Run the complete demonstration."""
    print("Starting Standardized Environment Configuration Demonstration...")
    
    try:
        demo_environment_layouts()
        demo_agent_parameters()
        demo_visualization_consistency()
        demo_method_integration()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("The standardized environment configuration is now ready for use!")
        print("All methods can now use the same environment layouts and visualization parameters.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 