import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines

gym.logger.set_level(40)
os.environ["GYM_CONFIG_CLASS"] = "Example"
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs import test_cases as tc


def main():
    """
    Minimum working example:
    2 agents: 1 running external policy, 1 running GA3C-CADRL
    """

    # Create single tf session for all experiments
    import tensorflow.compat.v1 as tf

    tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()

    # Instantiate the environment
    env = gym.make("CollisionAvoidance-v0")

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__))
        + "/../../experiments/results/example/"
    )

    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = tc.get_doorway()
    [
        agent.policy.initialize_network()
        for agent in agents
        if hasattr(agent.policy, "initialize_network")
    ]
    env.set_agents(agents)

    # Debug: Print agent attributes
    print("Agent attributes:", dir(agents[0]))
    print("Agent position:", agents[0].pos_global_frame)  # Changed from pos to pos_global_frame
    print("Agent goal:", agents[0].goal_global_frame)     # Changed from state to goal_global_frame

    obs = env.reset()  # Get agents' initial observations

    # Define number of simulation steps
    num_steps = 100

    # Store positions for animation
    positions_history = []
    for i in range(num_steps):
        # Query the external agents' policies
        actions = {}
        actions[0] = np.array([1.0, 0.5])

        # Run a simulation step
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

    # Example: Define obstacle positions (vertical wall at x=-2)
    obstacle_positions = [[-2, y] for y in range(-1, 2)]

    # Save the animation as a GIF
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True)

    # Separate agents and obstacles based on a property (e.g., policy or id)
    # For this example, let's assume the last agent is the obstacle (customize as needed)
    obstacle_indices = [i for i, agent in enumerate(agents) if hasattr(agent.policy, 'str') and agent.policy.str == 'Static']
    agent_indices = [i for i in range(len(agents)) if i not in obstacle_indices]

    # Create scatter plots for agents (blue circles) and obstacles
    agent_scatter = ax.scatter([], [], c='blue', s=100, label='Agent')
    obstacle_scatter = ax.scatter([], [], c='blue', s=100)  # Removed label for obstacle
    goal_scatters = []

    # Create goal markers for agents only (not obstacles)
    for i, agent in enumerate(agents):
        if i in agent_indices:
            goal = agent.goal_global_frame
            ax.plot(goal[0], goal[1], 'g*', markersize=10, label='Goal' if i == 0 else "")

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
        return agent_scatter, obstacle_scatter, *goal_scatters

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=100, blit=True)
    from pathlib import Path
    root_dir = Path(__file__).resolve().parents[5]
    animations_dir = root_dir / 'logs' / 'Social-CADRL' / 'animations'
    animations_dir.mkdir(parents=True, exist_ok=True)
    outfile = animations_dir / 'doorway_animation.gif'
    ani.save(str(outfile), writer='pillow')
    print('Animation generated: {}'.format(outfile))

    return True


if __name__ == "__main__":
    main()
    print("Experiment over.")
