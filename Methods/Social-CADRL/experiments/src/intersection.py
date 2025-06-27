import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
from gym import spaces
from matplotlib.legend_handler import HandlerTuple

# Disable observation space checking
gym.logger.set_level(40)
os.environ["GYM_CONFIG_CLASS"] = "Example"
os.environ["GYM_DISABLE_OBSERVATION_SPACE_CHECK"] = "1"

from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs import test_cases as tc

# Set the max number of agents to 26 before creating the environment
Config.MAX_NUM_AGENTS_IN_ENVIRONMENT = 26

def main():
    """
    Minimum working example:
    2 agents: 1 running external policy, 1 running GA3C-CADRL
    """

    # Create single tf session for all experiments
    import tensorflow.compat.v1 as tf

    tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()

    # Create the environment
    env = gym.make("CollisionAvoidance-v0")
    
    # Remove manual override of observation space
    # The environment's observation space will now match the config

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__))
        + "/../../experiments/results/example/"
    )

    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = tc.get_intersection()
    [
        agent.policy.initialize_network()
        for agent in agents
        if hasattr(agent.policy, "initialize_network")
    ]
    env.set_agents(agents)

    obs = env.reset()  # Get agents' initial observations

    # Define number of simulation steps
    num_steps = 100

    # Store positions for animation
    positions_history = []
    for i in range(num_steps):
        # Query the external agents' policies
        actions = {}
        actions[0] = np.array([1.0, 0.5])

        # Internal agents (running a pre-learned policy defined in envs/policies)
        # will automatically query their policy during env.step
        # ==> no need to supply actions for internal agents here

        # Run a simulation step (check for collisions, move sim agents)
        obs, rewards, terminated, truncated, which_agents_done = env.step(actions)
        
        # Store current positions
        current_positions = []
        for agent in agents:
            current_positions.append([agent.pos_global_frame[0], agent.pos_global_frame[1]])
        positions_history.append(current_positions)

        if terminated:
            print("All agents finished!")
            break
    env.reset()

    # Example: Define obstacle positions (horizontal wall at y=0)
    obstacle_positions = [[x, 0] for x in range(-2, 3)]

    # Save the animation as a GIF
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True)

    # Create scatter plots for agents and obstacles
    agent_scatter = ax.scatter([], [], c='blue', s=100, label='Agent')
    obstacle_scatter = ax.scatter([], [], c='blue', s=100)  # Removed label for obstacle
    goal_scatters = []

    # Create goal markers
    for agent in agents:
        goal = agent.goal_global_frame
        ax.plot(goal[0], goal[1], 'g*', markersize=10, label='Goal' if agent.id == 0 else "")

    # Custom legend handles
    agent_handle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='Agent')
    goal_handle = mlines.Line2D([], [], color='green', marker='*', linestyle='None', markersize=8, label='Goal')

    # Place legend outside the plot with smaller font
    ax.legend(
        [agent_handle, goal_handle],
        ['Agent', 'Goal'],
        loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, borderaxespad=0.
    )

    # Animation update function
    def update(frame):
        if frame < len(positions_history):
            agent_positions = []
            obstacle_positions = []
            goal_positions = []
            for i, agent in enumerate(agents):
                pos = positions_history[frame][i]
                is_obstacle = np.allclose(pos, agent.goal_global_frame)
                if is_obstacle:
                    obstacle_positions.append(pos)
                else:
                    agent_positions.append(pos)
                    goal_positions.append(agent.goal_global_frame)
            agent_scatter.set_offsets(np.array(agent_positions).reshape(-1, 2))
            obstacle_scatter.set_offsets(np.array(obstacle_positions).reshape(-1, 2))
            # Remove previous goal scatters
            for gs in goal_scatters:
                gs.remove()
            goal_scatters.clear()
            # Plot green stars for agent goals (not for obstacles)
            for goal in goal_positions:
                gs = ax.plot(goal[0], goal[1], 'g*', markersize=10, label=None)[0]
                goal_scatters.append(gs)
            # Add green stars on top of obstacles
            for pos in obstacle_positions:
                gs = ax.plot(pos[0], pos[1], 'g*', markersize=10, label=None)[0]
                goal_scatters.append(gs)
        return agent_scatter, obstacle_scatter, *goal_scatters

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100, blit=True)
    ani.save('intersection_animation.gif', writer='pillow')
    print('Animation generated: {}'.format(os.path.abspath('intersection_animation.gif')))

    return True


if __name__ == "__main__":
    main()
    print("Experiment over.")