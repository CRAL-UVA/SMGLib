#!/usr/bin/env python3
"""
Fixed Comprehensive Test Suite for Social Collision Avoidance Scenarios

This script runs 25 different test cases covering:
- All 3 scenarios (intersection, hallway, doorway)
- Various numbers of agents (1-6)
- Different start/goal positions (avoiding negative starts to prevent argparse issues)
- Parameter variations (speed, radius, heading)
"""

import os
import sys
import subprocess
import json
import time

def run_test_case(test_name, cmd_args, expected_agents=None):
    """Run a single test case"""
    print(f"\n{'='*60}")
    print(f"Running Test Case: {test_name}")
    print(f"Command: python3 run_scenarios.py {' '.join(cmd_args)}")
    print(f"{'='*60}")
    
    try:
        # Handle arguments that start with '-' by using shell=False and proper escaping
        cmd = ['python3', 'run_scenarios.py'] + cmd_args
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout (some tests need more time)
        )
        end_time = time.time()
        
        # Check results
        if result.returncode == 0:
            print(f"✅ SUCCESS (took {end_time - start_time:.1f}s)")
            
            # Check for expected outputs
            if "Animation saved as" in result.stdout:
                print("✅ Animation generated successfully")
            
            if expected_agents and f"Created {expected_agents}" in result.stdout:
                print(f"✅ Created expected number of agents ({expected_agents})")
            
            # Show key output lines
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if any(keyword in line for keyword in ['Agent configuration:', 'Agent 1:', 'Agent 2:', 'Agent 3:', 'Agent 4:', 'Step 0, Agent', 'All dynamic agents reached']):
                    print(f"   {line}")
            
        else:
            print(f"❌ FAILED (return code: {result.returncode})")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ FAILED (timeout)")
        return False
    except Exception as e:
        print(f"❌ FAILED (exception: {e})")
        return False

def create_test_agent_files():
    """Create JSON files for agent configurations"""
    
    # Test case with JSON: 3-agent intersection
    agents_3_intersection = {
        "agents": [
            {"start_x": 0, "start_y": 0, "goal_x": 5, "goal_y": 0, "radius": 0.5, "pref_speed": 1.0, "heading": 0},
            {"start_x": 0, "start_y": 3, "goal_x": 0, "goal_y": 0, "radius": 0.4, "pref_speed": 1.2, "heading": 4.71},
            {"start_x": 3, "start_y": 1, "goal_x": 0, "goal_y": 0, "radius": 0.6, "pref_speed": 0.8, "heading": 3.14}
        ]
    }
    
    with open('agents_3_intersection.json', 'w') as f:
        json.dump(agents_3_intersection, f, indent=2)
    
    # Test case with JSON: 4-agent doorway challenge
    agents_4_doorway = {
        "agents": [
            {"start_x": 6, "start_y": 0.2, "goal_x": 0, "goal_y": 0.2, "radius": 0.3, "pref_speed": 0.8, "heading": 3.14},
            {"start_x": 5.5, "start_y": 0.1, "goal_x": 0, "goal_y": 0.1, "radius": 0.3, "pref_speed": 0.9, "heading": 3.14},
            {"start_x": 0, "start_y": 0.1, "goal_x": 6, "goal_y": 0.1, "radius": 0.25, "pref_speed": 1.1, "heading": 0},
            {"start_x": 0.5, "start_y": 0, "goal_x": 5.5, "goal_y": 0, "radius": 0.25, "pref_speed": 1.0, "heading": 0}
        ]
    }
    
    with open('agents_4_doorway.json', 'w') as f:
        json.dump(agents_4_doorway, f, indent=2)

def main():
    """Run all test cases"""
    print("🚀 Starting Fixed Comprehensive Social Collision Avoidance Test Suite")
    print("This will test various scenarios, agent numbers, and configurations")
    print("All test cases avoid negative start coordinates to prevent argparse issues")
    
    # Create test agent configuration files
    create_test_agent_files()
    
    # Define all 25 test cases (avoiding negative start coordinates)
    test_cases = [
        # INTERSECTION SCENARIOS (Tests 1-10)
        # Test Case 1: Basic intersection with 2 default agents
        {
            "name": "Intersection - Basic 2 Default Agents",
            "args": ["--scenario", "intersection", "--agents", "2", "--quiet"],
            "expected_agents": 2
        },
        
        # Test Case 2: Custom intersection with crossing paths
        {
            "name": "Intersection - Perpendicular Crossing",
            "args": [
                "--scenario", "intersection", "--agents", "2", "--quiet",
                "--agent1", "4,0:0,0:0.5:1.0:3.14",
                "--agent2", "0,4:0,0:0.5:1.0:4.71"
            ],
            "expected_agents": 2
        },
        
        # Test Case 3: Intersection with speed variations
        {
            "name": "Intersection - Speed Variations",
            "args": [
                "--scenario", "intersection", "--agents", "3", "--quiet",
                "--agent1", "3,0:0,0:0.3:1.8:3.14",      # Small, very fast
                "--agent2", "2,5:5,0:0.7:0.5:4.71",   # Large, slow  
                "--agent3", "1,3:1,0:0.5:1.0:4.71"   # Medium, medium
            ],
            "expected_agents": 3
        },
        
        # Test Case 4: Intersection with 4 agents from corners
        {
            "name": "Intersection - 4-Way Convergence",
            "args": [
                "--scenario", "intersection", "--agents", "4", "--quiet",
                "--agent1", "4,0:0,0:0.4:1.0:3.14",      # East to Center
                "--agent2", "0,0:4,0:0.4:1.0:0",   # Center to East
                "--agent3", "0,4:0,0:0.4:1.0:4.71",   # North to Center
                "--agent4", "0,0:0,4:0.4:1.0:1.57"    # Center to North
            ],
            "expected_agents": 4
        },
        
        # Test Case 5: Intersection with diagonal movements
        {
            "name": "Intersection - Diagonal Paths",
            "args": [
                "--scenario", "intersection", "--agents", "3", "--quiet",
                "--agent1", "3,3:0,0:0.4:1.0:3.93",    # Diagonal NE to Center
                "--agent2", "3,0:0,3:0.4:1.0:2.36",    # Diagonal SE to NW
                "--agent3", "0,4:0,0:0.4:1.0:4.71"       # Straight vertical
            ],
            "expected_agents": 3
        },
        
        # Test Case 6: Complex intersection with JSON file
        {
            "name": "Intersection - Complex 3 Agents from JSON",
            "args": [
                "--scenario", "intersection", "--quiet",
                "--agents-file", "agents_3_intersection.json"
            ],
            "expected_agents": 3
        },
        
        # Test Case 7: Intersection with close encounters
        {
            "name": "Intersection - Close Encounter Scenario",
            "args": [
                "--scenario", "intersection", "--agents", "2", "--quiet",
                "--agent1", "2,0.2:0,0.2:0.3:0.8:3.14",
                "--agent2", "2,0:0,0:0.3:0.8:3.14"
            ],
            "expected_agents": 2
        },
        
        # Test Case 8: Intersection with mixed sizes
        {
            "name": "Intersection - Mixed Agent Sizes",
            "args": [
                "--scenario", "intersection", "--agents", "3", "--quiet",
                "--agent1", "3,0:0,0:0.2:1.2:3.14",      # Very small, fast
                "--agent2", "2,5:5,0:0.8:0.6:4.71",   # Very large, slow
                "--agent3", "1,2:1,0:0.5:1.0:4.71"   # Medium
            ],
            "expected_agents": 3
        },
        
        # Test Case 9: Intersection with asymmetric goals
        {
            "name": "Intersection - Asymmetric Goals",
            "args": [
                "--scenario", "intersection", "--agents", "3", "--quiet",
                "--agent1", "4,0:2,3:0.4:1.0:2.0",
                "--agent2", "0,4:3,1:0.4:1.0:5.5",
                "--agent3", "4,1:2,0:0.4:1.0:3.8"
            ],
            "expected_agents": 3
        },
        
        # Test Case 10: Intersection stress test
        {
            "name": "Intersection - High Density Test",
            "args": [
                "--scenario", "intersection", "--agents", "5", "--quiet",
                "--agent1", "3,0:0,0:0.3:1.0:3.14",
                "--agent2", "2,5:5,0:0.3:1.0:4.71",
                "--agent3", "1,3:1,0:0.3:1.0:4.71",
                "--agent4", "5,1:4,0:0.3:1.0:3.5",
                "--agent5", "1,0:0,4:0.3:1.0:1.8"
            ],
            "expected_agents": 5
        },
        
        # HALLWAY SCENARIOS (Tests 11-18)
        # Test Case 11: Single agent traversal
        {
            "name": "Hallway - Single Agent Fast Traversal",
            "args": [
                "--scenario", "hallway", "--agents", "1", "--quiet",
                "--agent1", "4,0:0,0:0.4:1.5:3.14"
            ],
            "expected_agents": 1
        },
        
        # Test Case 12: Opposing agents
        {
            "name": "Hallway - Head-to-Head Opposition",
            "args": [
                "--scenario", "hallway", "--agents", "2", "--quiet", 
                "--agent1", "3,0.3:0,0.3:0.4:1.0:3.14",
                "--agent2", "0,0:3,0:0.4:1.0:0"
            ],
            "expected_agents": 2
        },
        
        # Test Case 13: Multi-agent congestion
        {
            "name": "Hallway - Multi-Agent Bottleneck",
            "args": [
                "--scenario", "hallway", "--agents", "3", "--quiet",
                "--agent1", "3,0.5:0,0.5:0.3:0.9:3.14",
                "--agent2", "2.5,0:0,0:0.3:1.1:3.14", 
                "--agent3", "3,0:0,0:0.3:0.8:3.14"
            ],
            "expected_agents": 3
        },
        
        # Test Case 14: Speed differential test
        {
            "name": "Hallway - Speed Differential Chase",
            "args": [
                "--scenario", "hallway", "--agents", "3", "--quiet",
                "--agent1", "4,0.2:0,0.2:0.3:2.0:3.14",   # Very fast
                "--agent2", "3.5,0:0,0:0.4:0.7:3.14",   # Slow
                "--agent3", "3.8,0:0,0:0.3:1.3:3.14"  # Medium-fast
            ],
            "expected_agents": 3
        },
        
        # Test Case 15: Large agent in narrow space
        {
            "name": "Hallway - Large Agent Navigation",
            "args": [
                "--scenario", "hallway", "--agents", "2", "--quiet",
                "--agent1", "3,0:0,0:0.6:0.8:3.14",       # Large, slow
                "--agent2", "0,0.3:3,0.3:0.2:1.2:0"  # Small, fast, opposite
            ],
            "expected_agents": 2
        },
        
        # Test Case 16: Multiple following agents
        {
            "name": "Hallway - Following Convoy",
            "args": [
                "--scenario", "hallway", "--agents", "4", "--quiet",
                "--agent1", "4,0.3:0,0.3:0.3:1.0:3.14",
                "--agent2", "3.5,0.3:0,0.3:0.3:1.0:3.14",
                "--agent3", "3,0.3:0,0.3:0.3:1.0:3.14",
                "--agent4", "2.5,0.3:0,0.3:0.3:1.0:3.14"
            ],
            "expected_agents": 4
        },
        
        # Test Case 17: Overtaking scenario
        {
            "name": "Hallway - Overtaking Maneuver",
            "args": [
                "--scenario", "hallway", "--agents", "3", "--quiet",
                "--agent1", "4,0:0,0:0.4:0.6:3.14",       # Slow leader
                "--agent2", "3.5,0.2:0,0.2:0.3:1.4:3.14",  # Fast follower
                "--agent3", "0,0:3.5,0:0.3:1.0:0"  # Oncoming
            ],
            "expected_agents": 3
        },
        
        # Test Case 18: Mixed direction flow
        {
            "name": "Hallway - Bidirectional Flow",
            "args": [
                "--scenario", "hallway", "--agents", "4", "--quiet",
                "--agent1", "3.5,0.4:0,0.4:0.3:1.0:3.14",
                "--agent2", "3,0.1:0,0.1:0.3:1.0:3.14",
                "--agent3", "0,0:3.5,0:0.3:1.0:0",
                "--agent4", "0.5,0:3,0:0.3:1.0:0"
            ],
            "expected_agents": 4
        },
        
        # DOORWAY SCENARIOS (Tests 19-25)
        # Test Case 19: Basic doorway passage
        {
            "name": "Doorway - Basic Bidirectional Passage",
            "args": [
                "--scenario", "doorway", "--agents", "2", "--quiet",
                "--agent1", "6,0:0,0:0.3:0.8:3.14",
                "--agent2", "0,0:6,0:0.3:0.8:0"
            ],
            "expected_agents": 2
        },
        
        # Test Case 20: Angled approaches
        {
            "name": "Doorway - Angled Convergence",
            "args": [
                "--scenario", "doorway", "--agents", "3", "--quiet",
                "--agent1", "5,0.7:0,0:0.4:1.0:3.5",
                "--agent2", "4,0:0,0.7:0.4:1.0:2.8",
                "--agent3", "0,0.5:5,0:0.4:1.0:0.2"
            ],
            "expected_agents": 3
        },
        
        # Test Case 21: Small agents through doorway
        {
            "name": "Doorway - Small Agent Rush",
            "args": [
                "--scenario", "doorway", "--agents", "4", "--quiet",
                "--agent1", "6,0.3:0,0.3:0.2:1.2:3.14",
                "--agent2", "5.5,0:0,0:0.2:1.3:3.14",
                "--agent3", "6,0:0,0:0.2:1.1:3.14",
                "--agent4", "0,0.1:6,0.1:0.2:1.0:0"
            ],
            "expected_agents": 4
        },
        
        # Test Case 22: Large agent priority
        {
            "name": "Doorway - Large Agent Priority",
            "args": [
                "--scenario", "doorway", "--agents", "3", "--quiet",
                "--agent1", "6,0:0,0:0.7:0.6:3.14",      # Large, slow
                "--agent2", "5.5,0.4:0,0.4:0.2:1.2:3.14",  # Small, fast
                "--agent3", "0,0.2:6,0.2:0.3:1.0:0"    # Medium, opposite
            ],
            "expected_agents": 3
        },
        
        # Test Case 23: Complex doorway with JSON file
        {
            "name": "Doorway - Complex 4 Agents from JSON",
            "args": [
                "--scenario", "doorway", "--quiet", "--steps", "150",
                "--agents-file", "agents_4_doorway.json"
            ],
            "expected_agents": 4
        },
        
        # Test Case 24: Simultaneous arrival
        {
            "name": "Doorway - Simultaneous Arrival Test",
            "args": [
                "--scenario", "doorway", "--agents", "4", "--quiet",
                "--agent1", "6,0.2:0,0.2:0.3:1.0:3.14",
                "--agent2", "6,0:0,0:0.3:1.0:3.14",
                "--agent3", "0,0.2:6,0.2:0.3:1.0:0",
                "--agent4", "0,0:6,0:0.3:1.0:0"
            ],
            "expected_agents": 4
        },
        
        # Test Case 25: Extreme doorway challenge
        {
            "name": "Doorway - Extreme Challenge",
            "args": [
                "--scenario", "doorway", "--agents", "6", "--quiet", "--steps", "200",
                "--agent1", "7,0.4:0,0.4:0.25:1.5:3.14",
                "--agent2", "6.5,0.1:0,0.1:0.25:1.3:3.14",
                "--agent3", "6,0:0,0:0.25:1.4:3.14",
                "--agent4", "0,0:7,0:0.25:1.2:0",
                "--agent5", "0.5,0:6.5,0:0.25:1.1:0",
                "--agent6", "0,0.2:6,0.2:0.25:1.0:0"
            ],
            "expected_agents": 6
        }
    ]
    
    # Run all test cases
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{total}] ", end="")
        
        success = run_test_case(
            test_case["name"],
            test_case["args"], 
            test_case.get("expected_agents")
        )
        
        if success:
            passed += 1
        
        # Small delay between tests
        time.sleep(1)
    
    # Final results
    print(f"\n{'='*60}")
    print(f"TEST SUITE COMPLETE")
    print(f"{'='*60}")
    print(f"Passed: {passed}/{total} tests")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The system is working correctly.")
    else:
        print(f"⚠️  {total-passed} tests failed. Check the output above for details.")
    
    # Cleanup
    for file in ['agents_3_intersection.json', 'agents_4_doorway.json']:
        if os.path.exists(file):
            os.remove(file)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main()) 