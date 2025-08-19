import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
from pathlib import Path

# Import standardized environment configuration
sys.path.append(str(Path(__file__).resolve().parents[3] / 'src'))
from utils import StandardizedEnvironment

def plot_trajectory_standardized(agent_list, r_min, env_type='doorway'):
    """
    Plot trajectory using standardized environment configuration.
    
    Args:
        agent_list: List of agent objects with position and target attributes
        r_min: Minimum collision distance
        env_type: Environment type ('doorway', 'hallway', 'intersection')
    """
    
    # Create standardized plot
    fig, ax = StandardizedEnvironment.create_standard_plot(env_type, show_obstacles=True)
    
    # Plot each agent's trajectory
    for i, agent in enumerate(agent_list):
        # Determine if agent is stationary (obstacle)
        is_stationary = len(agent.position) <= 1 or all(np.allclose(agent.position[0], p) for p in agent.position)
        
        if is_stationary:
            # Plot stationary agents as gray obstacles
            for k in range(len(agent.position)):
                pos = (agent.position[k][0], agent.position[k][1])
                circle = Circle(pos, radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, 
                              edgecolor='black', facecolor='gray', alpha=0.7, zorder=1)
                ax.add_artist(circle)
        else:
            # Plot moving agents with trajectory
            color = StandardizedEnvironment.AGENT_COLORS[i % len(StandardizedEnvironment.AGENT_COLORS)]
            
            # Draw trajectory line
            if len(agent.position) > 1:
                positions_array = np.array(agent.position)
                ax.plot(positions_array[:, 0], positions_array[:, 1], 
                       color=color, linewidth=2, alpha=0.7)
            
            # Draw circles for each step in trajectory
            for k in range(len(agent.position)):
                pos = (agent.position[k][0], agent.position[k][1])
                circle = Circle(pos, radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, 
                              edgecolor='black', facecolor=color, alpha=0.8, zorder=1)
                ax.add_artist(circle)
            
            # Mark start position with square
            start_pos = agent.position[0]
            ax.scatter(start_pos[0], start_pos[1], marker='s', s=200, 
                      edgecolor='black', facecolor=color, zorder=3)
            
            # Mark goal position with star
            if hasattr(agent, 'target') and len(agent.target) == 2:
                ax.scatter(agent.target[0], agent.target[1], marker='*', s=300, 
                          edgecolor='black', facecolor=color, zorder=4)
    
    # Create standardized legend
    num_moving_agents = sum(1 for agent in agent_list 
                           if len(agent.position) > 1 and not all(np.allclose(agent.position[0], p) for p in agent.position))
    legend_handles, legend_labels = StandardizedEnvironment.create_standard_legend(num_moving_agents)
    ax.legend(legend_handles, legend_labels,
              loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12, borderaxespad=0., markerscale=1.2)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    
    # Save the plot
    plt.savefig('trajectory_standardized.svg', bbox_inches='tight', dpi=300)
    plt.savefig('trajectory_standardized.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Standardized trajectory plot saved as 'trajectory_standardized.svg' and 'trajectory_standardized.png'")

def plot_environment_comparison():
    """
    Create a comparison plot showing all three standardized environments.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    env_types = ['doorway', 'hallway', 'intersection']
    
    for i, env_type in enumerate(env_types):
        ax = axes[i]
        
        # Create standardized plot for each environment
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
        elif env_type == 'hallway':
            obstacles = StandardizedEnvironment.get_hallway_obstacles()
        elif env_type == 'intersection':
            obstacles = StandardizedEnvironment.get_intersection_obstacles()
        
        for obs in obstacles:
            circle = Circle(obs, radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, 
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
    
    plt.tight_layout()
    plt.savefig('environment_comparison.png', bbox_inches='tight', dpi=300)
    plt.savefig('environment_comparison.svg', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Environment comparison plot saved as 'environment_comparison.png' and 'environment_comparison.svg'")

def create_standardized_animation_frames(agent_list, env_type='doorway', num_moving_agents=None):
    """
    Create standardized animation frames for video generation.
    
    Args:
        agent_list: List of agent objects
        env_type: Environment type
        num_moving_agents: Number of moving agents (if None, inferred from agent_list)
    
    Returns:
        List of matplotlib figures for each frame
    """
    frames = []
    
    if not agent_list:
        return frames
    
    max_frames = max(len(a.position) for a in agent_list)
    
    for step in range(max_frames):
        fig, ax = StandardizedEnvironment.create_standard_plot(env_type, show_obstacles=True)
        
        # Plot each agent at current step
        for i, agent in enumerate(agent_list):
            # Determine if agent is moving
            is_moving = (num_moving_agents is None or i < num_moving_agents) and len(agent.position) > 1
            
            if is_moving:
                color = StandardizedEnvironment.AGENT_COLORS[i % len(StandardizedEnvironment.AGENT_COLORS)]
                
                # Get current position
                pos_index = min(step, len(agent.position) - 1)
                pos = agent.position[pos_index]
                
                # Draw trajectory up to current step
                if pos_index > 0:
                    past_positions = np.array(agent.position[:pos_index+1])
                    ax.plot(past_positions[:, 0], past_positions[:, 1], 
                           color=color, linewidth=2, alpha=0.7)
                
                # Draw current position
                circle = Circle(pos, radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, 
                              edgecolor='black', facecolor=color, alpha=0.8, zorder=3)
                ax.add_patch(circle)
                
                # Mark start position
                ax.scatter(agent.position[0][0], agent.position[0][1], marker='s', s=200, 
                          edgecolor='black', facecolor=color, zorder=3)
                
                # Mark goal position
                if hasattr(agent, 'target') and len(agent.target) == 2:
                    ax.scatter(agent.target[0], agent.target[1], marker='*', s=300, 
                              edgecolor='black', facecolor=color, zorder=4)
            else:
                # Stationary agent (obstacle)
                pos = agent.position[0] if len(agent.position) > 0 else [0, 0]
                circle = Circle(pos, radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, 
                              edgecolor='black', facecolor='gray', alpha=0.7, zorder=1)
                ax.add_patch(circle)
        
        frames.append(fig)
    
    return frames

if __name__ == "__main__":
    # Example usage
    print("Creating environment comparison plot...")
    plot_environment_comparison()
    print("Done!") 