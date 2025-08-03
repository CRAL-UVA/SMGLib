import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gym import spaces
import matplotlib.lines as mlines

# Disable observation space checking
gym.logger.set_level(40)
os.environ["GYM_CONFIG_CLASS"] = "Example"
os.environ["GYM_DISABLE_OBSERVATION_SPACE_CHECK"] = "1"

from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs import test_cases as tc


class ObservationSpaceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Create a new observation space with 26 keys
        self.observation_space = spaces.Dict({
            i: spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
            for i in range(26)
        })

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info


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
    env = ObservationSpaceWrapper(env)  # Wrap the environment

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__))
        + "/../../experiments/results/example/"
    )

    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = tc.get_hallway()
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

    # Example: Define obstacle positions (vertical wall at x=0)
    obstacle_positions = [[0, y] for y in range(-5, 6)]

    # Save the animation as a GIF
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True)

    # Create scatter plots for agents (blue circles) and obstacles
    agent_scatter = ax.scatter([], [], c='blue', s=100, label='Agent')
    obstacle_scatter = ax.scatter([], [], c='blue', s=100)  # Removed label for obstacle
    goal_scatters = []

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
    ani.save('hallway_animation.gif', writer='pillow')
    print('Animation generated: {}'.format(os.path.abspath('hallway_animation.gif')))

    return True


if __name__ == "__main__":
    main()
    print("Experiment over.")
