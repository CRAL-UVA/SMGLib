#!/usr/bin/env python3
"""
SMGLib - Social Multi-robot Navigation Library
Main entry point for running simulations.

This is a clean, modular version that replaces the monolithic run_simulation.py
while maintaining the exact same functionality.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import config
from src.methods import AVAILABLE_METHODS
from src.utils import print_simulation_results

def display_menu():
    """Display the main menu for algorithm selection."""
    print("\n" + "="*60)
    print("SMGLib - Social Multi-robot Navigation Library")
    print("="*60)
    print("Available algorithms:")
    for i, method in enumerate(AVAILABLE_METHODS, 1):
        print(f"{i}. {method}")
    print("="*60)

def get_user_input():
    """Get simulation parameters from user."""
    display_menu()
    
    # Get algorithm choice
    while True:
        try:
            choice = int(input(f"\nSelect algorithm (1-{len(AVAILABLE_METHODS)}): "))
            if 1 <= choice <= len(AVAILABLE_METHODS):
                selected_method = AVAILABLE_METHODS[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(AVAILABLE_METHODS)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Get number of robots
    while True:
        try:
            num_robots = int(input("Number of robots (default 2): ") or "2")
            if num_robots > 0:
                break
            else:
                print("Number of robots must be positive")
        except ValueError:
            print("Please enter a valid number")
    
    # Get environment type
    env_types = ["hallway", "doorway", "intersection"]
    print(f"\nEnvironment types: {', '.join(env_types)}")
    env_type = input("Environment type (default 'hallway'): ").strip() or "hallway"
    if env_type not in env_types:
        print(f"Unknown environment type '{env_type}', using 'hallway'")
        env_type = "hallway"
    
    return selected_method, num_robots, env_type

def run_simulation(method_name: str, num_robots: int, env_type: str):
    """Run simulation using the specified method."""
    print(f"\nRunning {method_name} simulation...")
    print(f"Robots: {num_robots}, Environment: {env_type}")
    
    # Update configuration
    config.set_param("num_robots", num_robots)
    config.set_param("env_type", env_type)
    
    # Create necessary directories
    config.create_directories(method_name)
    
    # Import and run the appropriate method
    if method_name == "Social-ORCA":
        from src.simulators.orca_simulator import run_orca_simulation
        return run_orca_simulation(num_robots, env_type)
    
    elif method_name == "Social-CADRL":
        from src.simulators.cadrl_simulator import run_cadrl_simulation
        return run_cadrl_simulation(num_robots, env_type)
    
    elif method_name == "Social-IMPC-DR":
        from src.simulators.impc_simulator import run_impc_simulation  
        return run_impc_simulation(num_robots, env_type)
    
    else:
        print(f"Error: Unknown method '{method_name}'")
        return None

def main():
    """Main function."""
    try:
        # Get user input
        method_name, num_robots, env_type = get_user_input()
        
        # Run simulation
        result = run_simulation(method_name, num_robots, env_type)
        
        if result:
            print(f"\n✅ {method_name} simulation completed successfully!")
            
            # Print results if available
            if isinstance(result, dict):
                makespan = result.get('makespan', 0)
                flow_rate = result.get('flow_rate', 0)
                completion_data = result.get('completion_data', [])
                
                print_simulation_results(method_name, num_robots, makespan, 
                                       flow_rate, completion_data)
        else:
            print(f"\n❌ {method_name} simulation failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 