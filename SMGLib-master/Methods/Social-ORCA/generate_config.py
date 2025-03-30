#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import os

def create_agent_element(agent_id, start_x, start_y, goal_x, goal_y):
    agent = ET.Element('agent')
    agent.set('id', str(agent_id))
    
    start = ET.SubElement(agent, 'start')
    start.set('x', str(start_x))
    start.set('y', str(start_y))
    
    goal = ET.SubElement(agent, 'goal')
    goal.set('x', str(goal_x))
    goal.set('y', str(goal_y))
    
    return agent

def generate_config(env_type, num_robots, robot_positions):
    root = ET.Element('root')
    
    # Add agents section
    agents = ET.SubElement(root, 'agents', {'number': str(num_robots), 'type': 'orca'})
    default_params = ET.SubElement(agents, 'default_parameters', {
        'size': '0.3',
        'movespeed': '1',
        'agentsmaxnum': str(num_robots),
        'timeboundary': '5.4',
        'sightradius': '3.0',
        'timeboundaryobst': '33'
    })
    
    # Add individual agents
    for i in range(num_robots):
        agent = ET.SubElement(agents, 'agent', {
            'id': str(i),
            'start.xr': str(robot_positions[i]['start_x']),
            'start.yr': str(robot_positions[i]['start_y']),
            'goal.xr': str(robot_positions[i]['goal_x']),
            'goal.yr': str(robot_positions[i]['goal_y'])
        })
    
    # Add map section
    map_elem = ET.SubElement(root, 'map')
    ET.SubElement(map_elem, 'width').text = '64'
    ET.SubElement(map_elem, 'height').text = '64'
    ET.SubElement(map_elem, 'cellsize').text = '1'
    
    # Add grid
    grid = ET.SubElement(map_elem, 'grid')
    for _ in range(64):  # 64x64 grid
        row = ET.SubElement(grid, 'row')
        row.text = '0 ' * 63 + '0'  # 64 zeros per row
    
    # Add obstacles based on environment type
    obstacles = ET.SubElement(root, 'obstacles', {'number': '2'})
    if env_type == 'hallway':
        # Add hallway walls
        obstacle1 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle1, 'vertex', {'xr': '0', 'yr': '31'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '0', 'yr': '32'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '63', 'yr': '31'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '63', 'yr': '32'})
        
        obstacle2 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle2, 'vertex', {'xr': '0', 'yr': '35'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '0', 'yr': '36'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '63', 'yr': '35'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '63', 'yr': '36'})
    
    elif env_type == 'doorway':
        # Add doorway walls
        obstacle1 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle1, 'vertex', {'xr': '30', 'yr': '0'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '31', 'yr': '0'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '30', 'yr': '30'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '31', 'yr': '30'})
        
        obstacle2 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle2, 'vertex', {'xr': '30', 'yr': '34'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '31', 'yr': '34'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '30', 'yr': '64'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '31', 'yr': '64'})
    
    elif env_type == 'intersection':
        # Add intersection walls
        obstacle1 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle1, 'vertex', {'xr': '30', 'yr': '0'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '31', 'yr': '0'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '30', 'yr': '30'})
        ET.SubElement(obstacle1, 'vertex', {'xr': '31', 'yr': '30'})
        
        obstacle2 = ET.SubElement(obstacles, 'obstacle')
        ET.SubElement(obstacle2, 'vertex', {'xr': '0', 'yr': '30'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '0', 'yr': '31'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '30', 'yr': '30'})
        ET.SubElement(obstacle2, 'vertex', {'xr': '30', 'yr': '31'})
    
    # Add algorithm section
    algorithm = ET.SubElement(root, 'algorithm')
    ET.SubElement(algorithm, 'searchtype').text = 'direct'
    ET.SubElement(algorithm, 'breakingties').text = '0'
    ET.SubElement(algorithm, 'allowsqueeze').text = 'false'
    ET.SubElement(algorithm, 'cutcorners').text = 'false'
    ET.SubElement(algorithm, 'hweight').text = '1'
    ET.SubElement(algorithm, 'timestep').text = '0.1'
    ET.SubElement(algorithm, 'delta').text = '0.1'
    
    # Create XML tree and save to file
    tree = ET.ElementTree(root)
    config_filename = f'configs/config_{env_type}_{num_robots}_robots.xml'
    tree.write(config_filename, encoding='utf-8', xml_declaration=True)
    print(f"Configuration saved to {config_filename}")
    return config_filename

def main():
    print("Welcome to the Social-ORCA Configuration Generator\n")
    print("Available environments:")
    print("1. doorway")
    print("2. hallway")
    print("3. intersection\n")
    
    while True:
        try:
            env_choice = int(input("Enter environment type (1-3): "))
            if env_choice in [1, 2, 3]:
                break
            print("Invalid choice! Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input! Please enter a number.")
    
    env_types = {1: 'doorway', 2: 'hallway', 3: 'intersection'}
    env_type = env_types[env_choice]
    
    while True:
        try:
            num_robots = int(input("Enter number of robots: "))
            if num_robots > 0:
                break
            print("Invalid number! Please enter a positive number.")
        except ValueError:
            print("Invalid input! Please enter a number.")
    
    robot_positions = []
    for i in range(num_robots):
        print(f"\nRobot {i+1} configuration:")
        while True:
            try:
                start_x = float(input("Starting X position: "))
                start_y = float(input("Starting Y position: "))
                goal_x = float(input("Goal X position: "))
                goal_y = float(input("Goal Y position: "))
                robot_positions.append({'start_x': start_x, 'start_y': start_y, 'goal_x': goal_x, 'goal_y': goal_y})
                break
            except ValueError:
                print("Invalid position values!")
    
    config_file = generate_config(env_type, num_robots, robot_positions)
    print(f"\nConfiguration file generated: {config_file}")
    print("You can now run the simulation using this configuration file.")

if __name__ == "__main__":
    main() 