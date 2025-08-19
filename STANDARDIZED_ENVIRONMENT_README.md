# Standardized Environment Configuration

This document describes the centralized environment configuration system that provides consistent grid sizes, obstacle layouts, and visualization parameters across all social navigation methods in SMGLib.

## Overview

The standardized environment configuration ensures that all methods (Social-CADRL, Social-IMPC-DR, Social-ORCA) use the same:
- Grid dimensions and coordinate systems
- Obstacle layouts for each environment type
- Agent parameters and colors
- Visualization styling and legends

## Key Components

### 1. Centralized Configuration (`src/utils.py`)

The `StandardizedEnvironment` class provides:

```python
# Grid dimensions (based on CADRL's larger grid for better visibility)
GRID_X_MIN = -6.0
GRID_X_MAX = 6.0
GRID_Y_MIN = -8.0
GRID_Y_MAX = 8.0

# Standard agent parameters
DEFAULT_AGENT_RADIUS = 0.5
DEFAULT_PREF_SPEED = 1.0
DEFAULT_COLLISION_DISTANCE = 0.3

# Standard colors for agents (matching CADRL's color scheme)
AGENT_COLORS = [
    [0.8500, 0.3250, 0.0980],  # orange
    [0.0, 0.4470, 0.7410],     # blue
    [0.4660, 0.6740, 0.1880],  # green
    # ... more colors
]
```

### 2. Environment Types

#### Doorway Environment
- **Layout**: Vertical wall at x=0 with gap from y=-2 to y=2
- **Standard Positions**: 
  - Agent 1: Start=(-3.0, 0.0), Goal=(3.0, 0.0)
  - Agent 2: Start=(3.0, 0.0), Goal=(-3.0, 0.0)

#### Hallway Environment
- **Layout**: Horizontal walls at y=-2 and y=2
- **Standard Positions**:
  - Agent 1: Start=(-4.0, 0.0), Goal=(4.0, 0.0)
  - Agent 2: Start=(4.0, 0.0), Goal=(-4.0, 0.0)

#### Intersection Environment
- **Layout**: + shaped intersection with corridors centered at (0,0)
- **Standard Positions**:
  - Agent 1: Start=(-4.0, 0.0), Goal=(4.0, 0.0)
  - Agent 2: Start=(0.0, -4.0), Goal=(0.0, 4.0)

## Usage

### For Social-IMPC-DR

Use the standardized version:
```bash
python app2_standardized.py doorway
python app2_standardized.py hallway
python app2_standardized.py intersection
```

### For Social-CADRL

Use the standardized test cases:
```python
from test_cases_standardized import get_testcase_standardized

# Get standardized test case for doorway environment
agents = get_testcase_standardized('doorway', num_agents=2)
```

### For Visualization

Use the standardized plotting functions:
```python
from utils import StandardizedEnvironment

# Create standardized plot
fig, ax = StandardizedEnvironment.create_standard_plot('doorway', show_obstacles=True)

# Plot agent trajectory
StandardizedEnvironment.plot_agent_trajectory(ax, positions, agent_id)

# Create standardized legend
legend_handles, legend_labels = StandardizedEnvironment.create_standard_legend(num_agents)
```

## Files Created/Modified

### New Files
1. `src/utils.py` - Added `StandardizedEnvironment` class
2. `src/methods/Social-IMPC-DR/app2_standardized.py` - Standardized IMPC-DR implementation
3. `src/methods/Social-IMPC-DR/plot_standardized.py` - Standardized plotting for IMPC-DR
4. `src/methods/Social-CADRL/envs/test_cases_standardized.py` - Standardized test cases for CADRL
5. `src/methods/Social-CADRL/envs/visualize_standardized.py` - Standardized visualization for CADRL
6. `demo_standardized_environments.py` - Demonstration script
7. `STANDARDIZED_ENVIRONMENT_README.md` - This documentation

### Modified Files
1. `run_simulation.py` - Updated to use standardized IMPC-DR implementation

## Benefits

### 1. Consistency
- All methods now use the same grid dimensions and coordinate systems
- Obstacle layouts are identical across methods
- Agent parameters are standardized

### 2. Comparability
- Easy to compare performance between different methods
- Same environment setup ensures fair comparison
- Consistent visualization makes results easier to interpret

### 3. Maintainability
- Centralized configuration reduces code duplication
- Changes to environment layouts only need to be made in one place
- Standardized parameters reduce configuration errors

### 4. Extensibility
- Easy to add new environment types
- Simple to modify agent parameters globally
- Consistent interface for new methods

## Running the Demo

To see the standardized environment configuration in action:

```bash
cd SMGLib
python demo_standardized_environments.py
```

This will create:
- `standardized_environments_demo.png` - Comparison of all three environments
- `visualization_consistency_demo.png` - Example of standardized visualization

## Migration Guide

### For Existing IMPC-DR Users
1. Use `app2_standardized.py` instead of `app2.py`
2. The interface remains the same, but now uses standardized environments
3. Animations will be saved with consistent styling

### For Existing CADRL Users
1. Use `test_cases_standardized.py` for consistent test cases
2. Use `visualize_standardized.py` for consistent visualization
3. The original files remain available for backward compatibility

### For New Users
1. Start with the standardized versions for consistency
2. Use the demo script to understand the environment layouts
3. Follow the examples in this README

## Future Work

1. **Social-ORCA Standardization**: Create standardized configuration generation for ORCA
2. **Additional Environments**: Add more environment types (e.g., maze, open space)
3. **Parameter Tuning**: Add methods for easy parameter adjustment
4. **Validation**: Add validation scripts to ensure consistency across methods

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the `src` directory is in your Python path
2. **Path Issues**: Use absolute paths when importing the standardized configuration
3. **Version Conflicts**: The standardized versions are designed to work with existing code

### Getting Help

If you encounter issues:
1. Check that all required files are present
2. Verify that the standardized configuration is properly imported
3. Run the demo script to test the setup
4. Check the original method implementations for reference

## Conclusion

The standardized environment configuration provides a solid foundation for consistent social navigation research. By using the centralized configuration, researchers can ensure that their comparisons between different methods are fair and meaningful.

The system is designed to be backward compatible, so existing code will continue to work while new development can take advantage of the standardized environment layouts and visualization parameters. 