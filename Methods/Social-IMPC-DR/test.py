import      SET
import      datetime
from        run          import *
from        others   import *
import      numpy        as np
import pickle
import copy
import os
import csv
import evaluate

def data_capture(a, b, c):
    data = {
        'pos_list': copy.copy(a),
        'position_list': copy.copy(b),
        'terminal_index_list': copy.copy(c)
    }
    return data

def initialize():
    agent_list=[]
    for i in range(SET.Num):
        agent_list+=[ uav(i,SET.ini_x[i],SET.ini_v[i],SET.target[i],SET.K) ]

    return agent_list

def PLAN( Num, ini_x, ini_v,target,r_min,epsilon,h,K,episodes, num_moving_drones=None):

    # os.sched_setaffinity(0,[0,1,2,3,4,5,6,7])
    
    SET.initialize_set(Num, ini_x, ini_v, target,r_min,epsilon,h,K,episodes)

    obj = {}

    ReachGoal=False

    episodes=SET.episodes
    
    agent_list=initialize()

    collect_data(agent_list)

    obj[0] = data_capture(SET.pos_list, SET.position_list, SET.terminal_index_list)

    velocity_data = {i: [] for i in range(Num)}
    path_data = {i: [] for i in range(Num)}

    if num_moving_drones is None:
        num_moving_drones = Num


    # Calculate nominal trajectories for each robot
    nominal_trajectories = {}
    for robot_id in range(num_moving_drones):
        initial_x, initial_y = ini_x[robot_id]  # Initial position
        final_x, final_y = target[robot_id]     # Final position
        nominal_trajectories[robot_id] = {
            'initial': (initial_x, initial_y),
            'final': (final_x, final_y),
            'step_size_x': (final_x - initial_x) / episodes,  # Step size in x direction
            'step_size_y': (final_y - initial_y) / episodes,  # Step size in y direction
        }

    # Track whether each robot has reached its target
    target_reached = [False] * num_moving_drones  # Initialize to False for all robots
    
    # Track when all robots reach their goals for make-span calculation
    all_goals_reached = False
    completion_step = episodes  # Default to full episodes if not all reach goals

    # the main loop
    start =datetime.datetime.now()
    end = start

    for i in range(1,episodes+1):
        end_last=end


        obstacle_list=get_obstacle_list(agent_list,SET.Num)

        # Separate moving and stationary agents
        moving_agents = agent_list[:num_moving_drones]
        stationary_agents = agent_list[num_moving_drones:]

        # run one step for moving agents only
        moving_agents = run_one_step(moving_agents, obstacle_list)

        # update the main agent_list
        agent_list = moving_agents + stationary_agents

        # print
        end = datetime.datetime.now()
        print("Step %s have finished, running time is %s"%(i,end-end_last))
    

        # Store velocity data
        for j, agent in enumerate(agent_list):
            if j < num_moving_drones:
                # Check if the robot has reached its target
                if not target_reached[j]:
                    distance_to_target = np.linalg.norm(agent.p - target[j])
                    # Use a strict threshold - robots must be very close to target
                    if distance_to_target < 0.02:  # 0.02 units threshold (very strict)
                        target_reached[j] = True
                        print(f"Robot {j} reached goal at step {i}, distance: {distance_to_target:.4f}, position: {agent.p}, target: {target[j]}")
                    
                    # Debug: Show progress for robots that are getting closer
                    if i % 20 == 0:  # Every 20 steps
                        print(f"Robot {j} progress at step {i}: distance to goal = {distance_to_target:.4f}")

                vx, vy = agent.v  # Extract vx and vy from the agent's velocity
                velocity_data[j].append([vx, vy])

                # Actual position
                if target_reached[j]:
                    # Freeze actual position at the target
                    px, py = target[j]
                else:
                    # Use the current position
                    px, py = agent.p[0], agent.p[1]

                # Nominal position (always calculated for all steps)
                nominal_x = nominal_trajectories[j]['initial'][0] + i * nominal_trajectories[j]['step_size_x']
                nominal_y = nominal_trajectories[j]['initial'][1] + i * nominal_trajectories[j]['step_size_y']
                # Append actual and nominal positions to path_data
                path_data[j].append([px, py, nominal_x, nominal_y])
            else: # For stationary agents
                vx, vy = agent.v
                velocity_data[j].append([vx, vy])
                px, py = agent.p[0], agent.p[1]
                path_data[j].append([px, py, px, py]) # nominal is same as actual

        # Check if all moving robots have reached their goals (for make-span calculation)
        if not all_goals_reached and all(target_reached[:num_moving_drones]):
            all_goals_reached = True
            completion_step = i
            print(f"All moving robots reached their goals at step {i}")

        collect_data(agent_list)

        obj[i] = data_capture(SET.pos_list, SET.position_list, SET.terminal_index_list)

       # if ReachGoal:        
            #break
            
        #ReachGoal=check_reach_target(agent_list)
    evaluate.evaluateMetrics()

    obj['goal'] = SET.target

    # Save velocity data to CSV files
    for robot_id, velocities in velocity_data.items():
        if num_moving_drones is not None and robot_id >= num_moving_drones:
            continue
        filename = f"avg_delta_velocity_robot_{robot_id}.csv"
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["vx", "vy"])  # Write header
            writer.writerows(velocities)  # Write velocity data

    print("Velocity CSV files saved.")

    # Save path deviation data to CSV files
    # Each file contains the actual and nominal positions for each robot
    for robot_id, positions in path_data.items():
        if num_moving_drones is not None and robot_id >= num_moving_drones:
            continue
        filename = f"path_deviation_robot_{robot_id}.csv"
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["px", "py", "nominal_x", "nominal_y"])  # Write header
            writer.writerows(positions)  # Write position data

    print("Path deviation CSV files saved.")
    
    # Final check: if completion_step is still episodes, it means no robots reached goals
    if completion_step == episodes:
        print(f"No robots reached their goals within the simulation time ({episodes} steps)")
    else:
        print(f"All robots reached goals at step {completion_step} out of {episodes} total steps")
    
    return obj, agent_list, completion_step
    
