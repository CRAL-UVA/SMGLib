# Social-CADRL Integration Guide

## Overview

Social-CADRL has been integrated into the main `run_simulation.py` system with **dependency isolation** to handle version conflicts between different navigation methods.

## The Problem

Different navigation methods in this repository have conflicting dependency requirements:

| Method | NumPy | TensorFlow | Gym | Special Dependencies |
|--------|-------|------------|-----|---------------------|
| **Main System** | ≥1.19.2 | Not required | ≥0.26.2 | - |
| **Social-IMPC-DR** | 2.2.3 | Not required | ≥0.26.2 | cvxpy, mosek |
| **Social-CADRL** | <1.20 (needs `bool8`) | 2.x required | 0.21.0 | gym-collision-avoidance |

## The Solution

### 1. Environment Isolation Strategy

Social-CADRL runs in an **isolated subprocess** with its own virtual environment:

```
SMGLib-master/
├── run_simulation.py          # Main entry point
├── setup_cadrl_env.py         # CADRL environment setup
├── Methods/
│   ├── Social-CADRL/
│   │   ├── venv/              # CADRL-specific virtual environment
│   │   ├── experiments/
│   │   └── envs/
│   ├── Social-ORCA/           # Uses main environment
│   └── Social-IMPC-DR/        # Uses main environment
└── venv/                      # Main virtual environment
```

### 2. Integration Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Main System   │    │  CADRL Adapter  │    │  CADRL Process  │
│  (run_simulation)│ →  │  (subprocess)   │ →  │ (isolated env)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
     Main env              Cross-boundary           CADRL env
   (newer numpy)           communication          (older numpy)
```

## Setup Instructions

### Automatic Setup (Recommended)

The CADRL environment is **automatically created** when you first select Social-CADRL:

```bash
python run_simulation.py
# Choose option 3 (Social-CADRL)
# Environment will be set up automatically on first use
```

### Manual Setup (Optional)

If you prefer to set up the environment beforehand:

```bash
python setup_cadrl_env.py
```

### Verify Setup

Check if your environment is CADRL-compatible:

```bash
python setup_cadrl_env.py --check
```

## Usage

### Through Main Interface

```bash
python run_simulation.py
```

1. Select **3. Social-CADRL**
2. Choose environment (doorway/hallway/intersection)
3. Set number of agents (2-4 recommended)
4. Choose default or custom positions

### Direct CADRL Usage

```bash
cd Methods/Social-CADRL/experiments/src
python run_scenarios.py --scenario doorway --agents 2
```

## How It Works

### 1. Method Selection
When user selects Social-CADRL, `run_social_cadrl()` is called.

### 2. Environment Check & Setup
- Check if `Methods/Social-CADRL/venv/` exists
- If not, automatically create virtual environment with compatible dependencies:
  - numpy==1.19.5 (has `bool8` attribute)
  - tensorflow==2.8.0, gym==0.21.0, etc.
- This happens seamlessly on first use

### 3. Configuration Generation
- User parameters are converted to CADRL format
- Configuration saved to `Methods/Social-CADRL/config.json`

### 4. Direct Execution
```python
# Switch to CADRL directory and run directly
os.chdir(cadrl_experiments_dir)
sys.argv = ["run_scenarios.py", "--scenario", env_type, ...]
import run_scenarios
run_scenarios.main()
```

### 5. Result Processing
- Animation files saved to `Methods/Social-CADRL/experiments/results/`
- Output captured and displayed to user

## Coordinate System Translation

Different methods use different coordinate systems:

| Method | Coordinate Range | Origin |
|--------|------------------|--------|
| **Social-ORCA** | 0-63 | Bottom-left |
| **Social-IMPC-DR** | Real-world scale | Varies |
| **Social-CADRL** | -10 to 10 | Center |

The integration automatically handles coordinate translation between systems.

## Troubleshooting

### "Module 'numpy' has no attribute 'bool8'"
- **Cause**: Using NumPy ≥1.20
- **Solution**: Run `python setup_cadrl_env.py` to create isolated environment

### "No module named 'gym_collision_avoidance'"
- **Cause**: Missing CADRL-specific dependency
- **Solution**: Install manually or use alternative CADRL implementation

### "CADRL simulation timed out"
- **Cause**: Long simulation or environment issues
- **Solution**: Reduce number of agents or check CADRL environment

### Animation Not Generated
- **Check**: `Methods/Social-CADRL/experiments/results/example/animations/`
- **Cause**: Matplotlib backend issues in subprocess
- **Solution**: Verify CADRL environment has proper display settings

## Benefits of This Approach

1. **No Dependency Conflicts**: Each method uses its optimal dependencies
2. **Easy Maintenance**: Methods can be updated independently
3. **Reliable Integration**: Subprocess isolation prevents crashes
4. **User-Friendly**: Single interface for all methods
5. **Extensible**: Easy to add more methods with different requirements

## Alternative Approaches Considered

1. **Conda Environments**: More complex setup, platform-dependent
2. **Docker Containers**: Heavier, requires Docker installation
3. **Conditional Imports**: Complex code, potential runtime conflicts
4. **Version Pinning**: Limits functionality of other methods

The subprocess isolation approach was chosen for simplicity and reliability.

## Future Improvements

1. **Async Execution**: Run multiple methods simultaneously
2. **Result Standardization**: Common output format across methods
3. **Performance Comparison**: Automated benchmarking
4. **GUI Interface**: Visual configuration and monitoring 