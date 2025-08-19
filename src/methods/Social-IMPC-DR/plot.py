import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
from pathlib import Path

# Import standardized environment configuration
sys.path.append(str(Path(__file__).resolve().parents[3] / 'src'))
from utils import StandardizedEnvironment

color=[ '#1f77b4',
        '#ff7f0e', 
        '#2ca02c', 
        '#d62728',
        '#9467bd',
        '#8c564b', 
        '#e377c2',
        '#7f7f7f', 
        '#bcbd22',
        '#17becf',
        '#2F4F4F',
        '#CD5C5C',
        '#ADD8E6',
        '#663399',
        '#8FBC8F',
        '#00CED1',
        '#6A5ACD',
        '#808000',
        '#A0522D',
        '#FF4500',
        '#708090',
        '#BDB76B',
        '#FF6347',
        '#E9967A',
        '#F5DEB3',
        '#FFB6C1',
        '#556B2F',
        '#008080',
        '#7FFF00',
        '#FFA500',
        '#FF8C00',
        '#00FF7F',
        '#C0C0C0',
        '#483D8B',
        '#F08080',
        '#D3D3D3',
        '#66CDAA',
        '#FA8072',
        '#F4A460',
        '#48D1CC',
        '#8A2BE2',
        '#2E8B57']


# plot
def plot_trajectory(agent_list,r_min, env_type='doorway'):

    fig=plt.figure(figsize=StandardizedEnvironment.FIG_SIZE)
    axes = fig.add_subplot(111, aspect='equal')

    # Use standardized colors
    moving_colors = StandardizedEnvironment.AGENT_COLORS

    # First, draw obstacles (static agents) as gray circles
    for i, agent in enumerate(agent_list):
        # Determine if agent is stationary (obstacle)
        is_stationary = len(agent.position) <= 1 or all(np.allclose(agent.position[0], p) for p in agent.position)
        
        if is_stationary:
            # Draw obstacle as gray circle
            for k in range(len(agent.position)):
                pos=(agent.position[k][0],agent.position[k][1])
                c=Circle(pos, radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, 
                        edgecolor='black', facecolor='gray', alpha=0.7, zorder=1)
                axes.add_artist(c)
    
    # Then, draw dynamic agents with colors and trajectories
    for i, agent in enumerate(agent_list):
        # Determine if agent is stationary (obstacle)
        is_stationary = len(agent.position) <= 1 or all(np.allclose(agent.position[0], p) for p in agent.position)
        
        if not is_stationary:
            agent_color = moving_colors[i % len(moving_colors)]
            
            # Draw trajectory line
            if len(agent.position) > 1:
                positions_array = np.array(agent.position)
                axes.plot(positions_array[:, 0], positions_array[:, 1], 
                         color=agent_color, linewidth=2, alpha=0.7, zorder=2)
            
            # Draw circles for each step in this agent's trajectory
            for k in range(len(agent.position)):
                pos=(agent.position[k][0],agent.position[k][1])
                c=Circle(pos, radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, 
                        edgecolor='black', facecolor=agent_color, alpha=0.8, zorder=2)
                axes.add_artist(c)
            
            # Mark start position with square
            axes.scatter(agent.position[0][0], agent.position[0][1], marker='s', s=200, 
                        edgecolor='black', facecolor=agent_color, zorder=3)
            
            # Mark goal with green star
            axes.scatter(agent.target[0], agent.target[1], marker='*', s=300, 
                        edgecolor='black', facecolor='green', zorder=4)
    
    
    # Use standardized grid limits
    axes.set_xlim([StandardizedEnvironment.GRID_X_MIN, StandardizedEnvironment.GRID_X_MAX])
    axes.set_ylim([StandardizedEnvironment.GRID_Y_MIN, StandardizedEnvironment.GRID_Y_MAX])
    axes.set_xlabel('x(m)')
    axes.set_ylabel('y(m)')

    plt.savefig('trajecotry.svg')
    plt.close()