# Social Collision Avoidance CLI Usage Guide

## Overview

The `run_scenarios.py` script now supports both interactive mode and command-line interface (CLI) mode for running social collision avoidance scenarios.

## Usage Modes

### 1. Interactive Mode (Original)
```bash
python3 run_scenarios.py
```
- Prompts you for scenario type, number of agents, and agent parameters
- Good for quick testing and exploration

### 2. CLI Mode (New)
```bash
python3 run_scenarios.py --scenario <type> --agents <num> [options]
```
- Specify all parameters via command line arguments
- Perfect for automation, testing, and scripting

## CLI Arguments

### Basic Arguments
- `--scenario, -s`: Scenario type (`intersection`, `hallway`, `doorway`)
- `--agents, -n`: Number of agents to create
- `--steps`: Maximum simulation steps (default: 100)
- `--quiet, -q`: Reduce output verbosity
- `--no-animation`: Skip animation generation
- `--output-dir, -o`: Output directory for animations

### Agent Configuration
- `--agent1`: First agent parameters
- `--agent2`: Second agent parameters
- ... up to `--agent10`: Tenth agent parameters
- `--agents-file`: Load agents from JSON file

### Agent Parameter Format
```
"start_x,start_y:goal_x,goal_y:radius:pref_speed:heading"
```

**Examples:**
- `"-3,0:5,0:0.5:1.0:0"` - Start at (-3,0), goal at (5,0), radius 0.5, speed 1.0, heading 0
- `"2,-5:5,0:0.4:1.2:1.57"` - Start at (2,-5), goal at (5,0), radius 0.4, speed 1.2, heading 1.57 (90°)

**Optional parameters:** If not specified, defaults are used based on scenario type.

## Example Commands

### Basic Examples

```bash
# Simple intersection with 2 default agents
python3 run_scenarios.py --scenario intersection --agents 2

# Hallway with 1 custom agent
python3 run_scenarios.py --scenario hallway --agents 1 --agent1 "-4,0:4,0:0.4:1.2:0"

# Doorway with opposing agents
python3 run_scenarios.py --scenario doorway --agents 2 \
  --agent1 "-6,0:6,0:0.3:0.8:0" \
  --agent2 "6,0:-6,0:0.3:0.8:3.14"
```

### Advanced Examples

```bash
# Intersection with 3 agents, custom properties
python3 run_scenarios.py --scenario intersection --agents 3 \
  --agent1 "-3,0:5,0:0.3:1.5:0" \
  --agent2 "2,-5:5,0:0.7:0.7:1.57" \
  --agent3 "-1,3:1,-3:0.5:1.0:4.71" \
  --steps 150

# Load agents from JSON file
python3 run_scenarios.py --scenario intersection --agents-file my_agents.json

# Quiet mode with no animation
python3 run_scenarios.py --scenario hallway --agents 2 --quiet --no-animation
```

## JSON Agent Configuration

Create a JSON file with agent configurations:

```json
{
  "agents": [
    {
      "start_x": -3,
      "start_y": 0,
      "goal_x": 5,
      "goal_y": 0,
      "radius": 0.5,
      "pref_speed": 1.0,
      "heading": 0
    },
    {
      "start_x": 0,
      "start_y": -3,
      "goal_x": 0,
      "goal_y": 5,
      "radius": 0.4,
      "pref_speed": 1.2,
      "heading": 1.57
    }
  ]
}
```

Then use:
```bash
python3 run_scenarios.py --scenario intersection --agents-file agents.json
```

## Test Suite

Run the comprehensive test suite with 10 different test cases:

```bash
python3 test_suite.py
```

### Test Cases Included:

1. **Basic Intersection** - 2 default agents
2. **Custom Intersection** - Crossing paths with specific positions
3. **Hallway Traversal** - Single agent navigation
4. **Hallway Opposition** - Two agents passing each other
5. **Doorway Passage** - Basic bidirectional flow
6. **Complex Intersection** - 3 agents from JSON file
7. **Varied Properties** - Different speeds and sizes
8. **Hallway Congestion** - Multiple agents in narrow space
9. **Angled Approaches** - Non-straight paths through doorway
10. **Doorway Challenge** - 4 agents from JSON file

## Parameter Ranges

### Typical Values:
- **Radius**: 0.25 - 0.7 meters
- **Preferred Speed**: 0.5 - 2.0 m/s
- **Heading**: 0 - 6.28 radians (0 - 360°)

### Scenario-Specific Ranges:
- **Intersection**: Start positions around (-5,-5) to (5,5)
- **Hallway**: Start positions along x-axis from -4 to 4
- **Doorway**: Start positions from -6 to 6, goals opposite side

## Common Heading Values:
- `0` - East (positive x)
- `1.57` - North (positive y) 
- `3.14` - West (negative x)
- `4.71` - South (negative y)

## Troubleshooting

### Common Issues:

1. **Invalid agent string**: Check the format `"x,y:x,y:r:s:h"`
2. **JSON file not found**: Ensure the file path is correct
3. **Animation fails**: Use `--no-animation` to skip visualization
4. **Long runtime**: Reduce `--steps` or use `--quiet` mode

### Getting Help:
```bash
python3 run_scenarios.py --help
```

## Performance Tips

- Use `--quiet` for faster execution
- Use `--no-animation` when testing multiple scenarios
- Limit `--steps` for quicker results
- Run test suite to validate functionality: `python3 test_suite.py` 