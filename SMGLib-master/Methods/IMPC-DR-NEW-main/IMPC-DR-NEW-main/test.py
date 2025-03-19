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

def PLAN( Num, ini_x, ini_v,target,r_min,epsilon,h,K,episodes):

    # os.sched_setaffinity(0,[0,1,2,3,4,5,6,7])
    
    SET.initialize_set(Num, ini_x , ini_v ,target,r_min,epsilon ,h ,K,episodes)

    obj = {}

    ReachGoal=False

    episodes=SET.episodes
    
    agent_list=initialize()

    collect_data(agent_list)

    obj[0] = data_capture(SET.pos_list, SET.position_list, SET.terminal_index_list)

    velocity_data = {i: [] for i in range(Num)}
    path_data = {i: [] for i in range(Num)}


    # Calculate nominal trajectories for each robot
    nominal_trajectories = {}
    for robot_id in range(Num):
        initial_x, initial_y = ini_x[robot_id]  # Initial position
        final_x, final_y = target[robot_id]     # Final position
        nominal_trajectories[robot_id] = {
            'initial': (initial_x, initial_y),
            'final': (final_x, final_y),
            'step_size_x': (final_x - initial_x) / episodes,  # Step size in x direction
            'step_size_y': (final_y - initial_y) / episodes,  # Step size in y direction
        }

    # Track whether each robot has reached its target
    target_reached = [False] * Num  # Initialize to False for all robots

    # the main loop
    start =datetime.datetime.now()
    end = start

    for i in range(1,episodes+1):
        end_last=end


        obstacle_list=get_obstacle_list(agent_list,SET.Num)

        # run one step
        agent_list = run_one_step(agent_list,obstacle_list)

        # print
        end = datetime.datetime.now()
        print("Step %s have finished, running time is %s"%(i,end-end_last))
    

        # Store velocity data
        for j, agent in enumerate(agent_list):
            # Check if the robot has reached its target
            if not target_reached[j]:
                distance_to_target = np.linalg.norm(agent.p - target[j])
                if distance_to_target < 0.01:  # Threshold for reaching the target
                    target_reached[j] = True

            vx, vy = agent.v  # Extract vx and vy from the agent's velocity
            velocity_data[j].append([vx, vy])

                        # Actual position
            if target_reached[j]:
                # Freeze actual position at the target
                px, py = target[j]
            else:
                # Use the current position
                px, py = agent.p[0], agent.p[1]

            #px, py = agent.p[0], agent.p[1]  
            #path_data[j].append([px, py])

            # Nominal position (always calculated for all steps)
            nominal_x = nominal_trajectories[j]['initial'][0] + i * nominal_trajectories[j]['step_size_x']
            nominal_y = nominal_trajectories[j]['initial'][1] + i * nominal_trajectories[j]['step_size_y']
            # Append actual and nominal positions to path_data
            path_data[j].append([px, py, nominal_x, nominal_y])

        collect_data(agent_list)

        obj[i] = data_capture(SET.pos_list, SET.position_list, SET.terminal_index_list)

       # if ReachGoal:        
            #break
            
        #ReachGoal=check_reach_target(agent_list)
    evaluate.evaluateMetrics()

    obj['goal'] = SET.target

    # Save velocity data to CSV files
    for robot_id, velocities in velocity_data.items():
        filename = f"avg_delta_velocity_robot_{robot_id}.csv"
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["vx", "vy"])  # Write header
            writer.writerows(velocities)  # Write velocity data

    print("Velocity CSV files saved.")

    # Save path deviation data to CSV files
    # Each file contains the actual and nominal positions for each robot
    for robot_id, positions in path_data.items():
        filename = f"path_deviation_robot_{robot_id}.csv"
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["px", "py", "nominal_x", "nominal_y"])  # Write header
            writer.writerows(positions)  # Write position data

    print("Path deviation CSV files saved.")
    
    
    return obj, agent_list
    
