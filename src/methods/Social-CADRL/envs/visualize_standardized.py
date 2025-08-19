"""
Standardized visualization for Social-CADRL using centralized environment configuration.
This module provides consistent visualization parameters across all methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib
import os
import glob
import imageio
import sys
from pathlib import Path

# Import standardized environment configuration
sys.path.append(str(Path(__file__).resolve().parents[4] / 'src'))
from utils import StandardizedEnvironment

matplotlib.rcParams.update({'font.size': 24})

def get_plot_save_dir_standardized(plot_save_dir, plot_policy_name, agents=None):
    """Get standardized plot save directory."""
    if plot_save_dir is None:
        plot_save_dir = os.path.dirname(os.path.realpath(__file__)) + '/../logs/test_cases/'
        os.makedirs(plot_save_dir, exist_ok=True)
    if plot_policy_name is None:
        plot_policy_name = agents[0].policy.str

    collision_plot_dir = plot_save_dir + "/collisions/"
    os.makedirs(collision_plot_dir, exist_ok=True)

    base_fig_name = "{test_case}_{policy}_{num_agents}agents{step}.{extension}"
    return plot_save_dir, plot_policy_name, base_fig_name, collision_plot_dir

def animate_episode_standardized(num_agents, plot_save_dir=None, plot_policy_name=None, test_case_index=0, agents=None, env_type='doorway'):
    """Animate episode using standardized environment configuration."""
    plot_save_dir, plot_policy_name, base_fig_name, collision_plot_dir = get_plot_save_dir_standardized(plot_save_dir, plot_policy_name, agents)

    # Load all images of the current episode (each animation)
    fig_name = base_fig_name.format(
            policy=plot_policy_name,
            num_agents = num_agents,
            test_case = str(test_case_index).zfill(3),
            step="_*",
            extension='png')
    last_fig_name = base_fig_name.format(
            policy=plot_policy_name,
            num_agents = num_agents,
            test_case = str(test_case_index).zfill(3),
            step="",
            extension='png')
    all_filenames = plot_save_dir+fig_name
    last_filename = plot_save_dir+last_fig_name

    # Dump all those images into a gif (sorted by timestep)
    filenames = glob.glob(all_filenames)
    filenames.sort()
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        os.remove(filename)
    for i in range(10):
        images.append(imageio.imread(last_filename))

    # Save the gif in a new animations sub-folder
    animation_filename = base_fig_name.format(
            policy=plot_policy_name,
            num_agents = num_agents,
            test_case = str(test_case_index).zfill(3),
            step="",
            extension='gif')
    animation_save_dir = plot_save_dir+"animations/"
    os.makedirs(animation_save_dir, exist_ok=True)
    animation_filename = animation_save_dir+animation_filename
    
    # Try imageio first, fall back to pillow if ffmpeg is not available
    try:
        imageio.mimsave(animation_filename, images, fps=StandardizedEnvironment.ANIMATION_FPS)
        print(f"Animation saved using imageio: {animation_filename}")
    except Exception as e:
        print(f"imageio failed ({e}), trying alternative method...")
        try:
            # Use PIL to create GIF as fallback
            from PIL import Image
            pil_images = []
            for img_array in images:
                # Convert numpy array to PIL Image
                if img_array.dtype != 'uint8':
                    img_array = (img_array * 255).astype('uint8')
                pil_img = Image.fromarray(img_array)
                pil_images.append(pil_img)
            
            # Save as GIF using PIL
            if pil_images:
                pil_images[0].save(
                    animation_filename,
                    save_all=True,
                    append_images=pil_images[1:],
                    duration=StandardizedEnvironment.ANIMATION_INTERVAL,  # milliseconds per frame
                    loop=0
                )
                print(f"Animation saved using PIL: {animation_filename}")
            else:
                print("No images to save for animation")
        except Exception as pil_error:
            print(f"PIL fallback also failed: {pil_error}")
            print("Animation could not be created, but simulation data is still available")

def plot_episode_standardized(agents, in_evaluate_mode, env_map=None, test_case_index=0, env_id=0,
    circles_along_traj=True, plot_save_dir=None, plot_policy_name=None,
    save_for_animation=False, limits=None, perturbed_obs=None,
    fig_size=None, show=False, save=False, env_type='doorway'):

    if max([agent.step_num for agent in agents]) == 0:
        return

    plot_save_dir, plot_policy_name, base_fig_name, collision_plot_dir = get_plot_save_dir_standardized(plot_save_dir, plot_policy_name, agents)

    # Use standardized figure size if not specified
    if fig_size is None:
        fig_size = StandardizedEnvironment.FIG_SIZE

    fig = plt.figure(env_id)
    fig.set_size_inches(fig_size[0], fig_size[1])

    plt.clf()

    ax = fig.add_subplot(1, 1, 1)

    if perturbed_obs is None:
        # Normal case of plotting
        max_time = draw_agents_standardized(agents, circles_along_traj, ax, env_type)
    else:
        max_time = draw_agents_standardized(agents, circles_along_traj, ax, env_type, last_index=-2)
        plot_perturbed_observation(agents, ax, perturbed_obs)

    # Label the axes
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    # plotting style (only show axis on bottom and left)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.draw()

    if limits is None:
        # Use standardized grid limits
        plt.xlim(StandardizedEnvironment.GRID_X_MIN, StandardizedEnvironment.GRID_X_MAX)
        plt.ylim(StandardizedEnvironment.GRID_Y_MIN, StandardizedEnvironment.GRID_Y_MAX)
        ax.set_aspect('equal')
    else:
        ax.axis('equal')

    if in_evaluate_mode and save:
        fig_name = base_fig_name.format(
            policy=plot_policy_name,
            num_agents = len(agents),
            test_case = str(test_case_index).zfill(3),
            step="",
            extension='png')
        filename = plot_save_dir+fig_name
        plt.savefig(filename)

        if np.any([agent.in_collision for agent in agents]):
            plt.savefig(collision_plot_dir+fig_name)

    if save_for_animation:
        fig_name = base_fig_name.format(
            policy=plot_policy_name,
            num_agents = len(agents),
            test_case = str(test_case_index).zfill(3),
            step="_"+"{:06.1f}".format(max_time),
            extension='png')
        filename = plot_save_dir+fig_name
        plt.savefig(filename)

    if show:
        plt.pause(0.0001)

def draw_agents_standardized(agents, circles_along_traj, ax, env_type='doorway', last_index=-1):
    """Draw agents using standardized environment configuration."""
    max_time = max([agent.global_state_history[agent.step_num+last_index, 0] for agent in agents] + [1e-4])
    max_time_alpha_scalar = 1.2
    
    for i, agent in enumerate(agents):
        # Use standardized colors
        color_ind = i % len(StandardizedEnvironment.AGENT_COLORS)
        plt_color = StandardizedEnvironment.AGENT_COLORS[color_ind]

        if circles_along_traj:
            plt.plot(agent.global_state_history[:agent.step_num+last_index+1, 1],
                     agent.global_state_history[:agent.step_num+last_index+1, 2],
                     color=plt_color, ls='-', linewidth=2)
            plt.plot(agent.global_state_history[0, 3],
                     agent.global_state_history[0, 4],
                     color=plt_color, marker='*', markersize=20)

            # Display circle at agent pos every circle_spacing (nom 1.5 sec)
            circle_spacing = 0.4
            circle_times = np.arange(0.0, agent.global_state_history[agent.step_num+last_index, 0],
                                     circle_spacing)
            _, circle_inds = find_nearest(agent.global_state_history[:agent.step_num, 0],
                                          circle_times)
            for ind in circle_inds:
                alpha = 1 - \
                        agent.global_state_history[ind, 0] / \
                        (max_time_alpha_scalar*max_time)
                c = rgba2rgb(plt_color+[float(alpha)])
                ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
                             radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, fc=c, ec=plt_color,
                             fill=True))

            # Display text of current timestamp every text_spacing (nom 1.5 sec)
            text_spacing = 1.5
            text_times = np.arange(0.0, agent.global_state_history[agent.step_num+last_index, 0],
                                   text_spacing)
            _, text_inds = find_nearest(agent.global_state_history[:agent.step_num, 0],
                                        text_times)
            for ind in text_inds:
                y_text_offset = 0.1
                alpha = agent.global_state_history[ind, 0] / \
                    (max_time_alpha_scalar*max_time)
                if alpha < 0.5:
                    alpha = 0.3
                else:
                    alpha = 0.9
                c = rgba2rgb(plt_color+[float(alpha)])
                ax.text(agent.global_state_history[ind, 1]-0.15,
                        agent.global_state_history[ind, 2]+y_text_offset,
                        '%.1f' % agent.global_state_history[ind, 0], color=c)
            
            # Also display circle at agent position at end of trajectory
            ind = agent.step_num + last_index
            alpha = 1 - \
                agent.global_state_history[ind, 0] / \
                (max_time_alpha_scalar*max_time)
            c = rgba2rgb(plt_color+[float(alpha)])
            ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
                         radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, fc=c, ec=plt_color))
            y_text_offset = 0.1
            ax.text(agent.global_state_history[ind, 1] - 0.15,
                    agent.global_state_history[ind, 2] + y_text_offset,
                    '%.1f' % agent.global_state_history[ind, 0],
                    color=plt_color)

        else:
            colors = np.zeros((agent.step_num, 4))
            colors[:,:3] = plt_color
            colors[:, 3] = np.linspace(0.2, 1., agent.step_num)
            colors = rgba2rgb(colors)

            plt.scatter(agent.global_state_history[:agent.step_num, 1],
                     agent.global_state_history[:agent.step_num, 2],
                     color=colors)

            # Also display circle at agent position at end of trajectory
            ind = agent.step_num + last_index
            alpha = 0.7
            c = rgba2rgb(plt_color+[float(alpha)])
            ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
                         radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS, fc=c, ec=plt_color))
    
    return max_time

def plot_perturbed_observation(agents, ax, perturbed_info):
    """Plot perturbed observation (keeping original implementation for compatibility)."""
    # This is hard-coded for 2 agent scenarios
    for i, agent in enumerate(agents):
        try:
            perturbed_obs = perturbed_info['perturbed_obs'][i]
        except:
            continue
        perturber = perturbed_info['perturber']
        other_agent_pos = agents[1].global_state_history[min(agent.step_num - 2, agents[1].step_num-1), 1:3]
        other_agent_perturbed_pos = agent.ego_pos_to_global_pos(perturbed_obs[4:6])
        rotation_angle = agent.ego_to_global_theta
        rotation_angle_deg = np.degrees(agent.ego_to_global_theta)
        other_agent_perturbed_lower_left_before_rotation = other_agent_perturbed_pos
        eps_lower_left_before_rotation = np.dot(np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]]), -perturber.epsilon_adversarial[0,4:6])
        other_agent_perturbed_lower_left_before_rotation = other_agent_perturbed_pos + eps_lower_left_before_rotation
        other_agent_lower_left_before_rotation = other_agent_pos + eps_lower_left_before_rotation
        ax.add_patch(plt.Circle(other_agent_perturbed_pos,
                     radius=agents[1].radius, fill=False, ec=StandardizedEnvironment.AGENT_COLORS[-1]))

        if perturber.p == "inf":
            ax.add_patch(plt.Rectangle(other_agent_perturbed_lower_left_before_rotation,
                width=2*perturber.epsilon_adversarial[0,4],
                height=2*perturber.epsilon_adversarial[0,5],
                angle=rotation_angle_deg,
                fill=False,
                linestyle='--'))
            ax.add_patch(plt.Rectangle(other_agent_lower_left_before_rotation,
                width=2*perturber.epsilon_adversarial[0,4],
                height=2*perturber.epsilon_adversarial[0,5],
                angle=rotation_angle_deg,
                fill=False,
                linestyle=':'))

        ps = agent.ego_pos_to_global_pos(perturber.perturbation_steps[:,0,4:6])

        perturb_colors = np.zeros((perturber.perturbation_steps.shape[0]-1, 4))
        perturb_colors[:,:3] = StandardizedEnvironment.AGENT_COLORS[-1]
        perturb_colors[:, 3] = np.linspace(0.2, 1.0, perturber.perturbation_steps.shape[0]-1)

        segs = np.reshape(np.hstack([ps[:-1], ps[1:]]), (perturber.perturbation_steps.shape[0]-1,2,2))[:-1]
        line_segments = LineCollection(segs, colors=perturb_colors, linestyle='solid')
        ax.add_collection(line_segments)

        plt.plot(other_agent_pos[0], other_agent_pos[1], 'x', color=StandardizedEnvironment.AGENT_COLORS[i+1], zorder=4)
        plt.plot(other_agent_perturbed_pos[0], other_agent_perturbed_pos[1], 'x', color=StandardizedEnvironment.AGENT_COLORS[-1], zorder=4)

# Helper functions (keeping original implementations for compatibility)
def find_nearest(array, value):
    """Find nearest value in array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def rgba2rgb(rgba):
    """Convert RGBA to RGB."""
    return rgba[:3]

def makedirs(path, exist_ok=False):
    """Create directories."""
    os.makedirs(path, exist_ok=exist_ok) 