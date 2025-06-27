# Flow Rate Implementation for Multi-Agent Navigation

## Overview
Flow rate is calculated using the formula: **Flow Rate = N / (z × T)**
Where:
- **N** = Number of agents
- **z** = Gap width (bottleneck width)
- **T** = Make-span (time for all agents to complete their tasks)

## Enhanced Make-Span Calculation

### ORCA Implementation
The make-span calculation now properly handles different completion scenarios:

1. **All agents reach goals**: Use the time when the last agent reaches its goal
2. **Some agents don't reach goals**: Use the total simulation time for fairness
3. **No agents reach goals**: Use the total simulation time

#### Detailed Logic:
- Extracts `pathfound="true/false"` and `steps` from ORCA log files
- Tracks individual agent completion times and success status
- Provides detailed reporting of which agents succeeded/failed
- Calculates flow rate based on successful agents only when appropriate

### Social-IMPC-DR Implementation
Enhanced to provide consistent reporting:

1. **Early completion**: All robots reach goals before max simulation steps
2. **Full simulation**: Uses max steps if robots don't reach goals early
3. **Individual tracking**: Already tracks robot goal completion accurately

## Gap Width Calculations

### ORCA (Grid Coordinates 0-64)
- **Doorway**: 4.0 grid units (gap at y=30-34)
- **Hallway**: 3.0 grid units (gap between walls at y=32-35)
- **Intersection**: 14.0 grid units (corridor width 25-39)

### Social-IMPC-DR (Normalized Coordinates 0-2.5)
- **Doorway**: 0.8 normalized units
- **Hallway**: 1.0 normalized units  
- **Intersection**: 0.8 normalized units

## Flow Rate Scenarios

### Scenario 1: All Agents Successful
```
Flow Rate = Total_Agents / (Gap_Width × Make_Span)
```

### Scenario 2: Partial Success
```
Flow Rate = Successful_Agents / (Gap_Width × Total_Simulation_Time)
```

### Scenario 3: No Success
```
Flow Rate = 0.0 (or minimal value based on attempted passage)
```

## Output Format

Both implementations now provide detailed output:
```
*****************************************************************
ORCA Flow Rate Calculation:
Scenario: All agents reached goals
Total agents: 2
Gap width (z): 4.0 grid units
Make-span (T): 15.30s
Flow Rate: 0.0327 agents/(unit·s)
*****************************************************************
```

## Key Improvements

1. **Accurate Goal Tracking**: Distinguishes between robots that reach goals vs. those that don't
2. **Fair Make-Span**: Uses appropriate time based on completion status
3. **Detailed Reporting**: Shows individual completion times and success rates
4. **Consistent Logic**: Both ORCA and IMPC-DR use similar enhancement patterns
5. **Robust Fallbacks**: Handles missing data gracefully

This implementation ensures that flow rate calculations accurately reflect the actual performance of navigation algorithms in different scenarios. 