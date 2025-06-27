#!/usr/bin/env python3
"""
Setup script for Social-CADRL environment with dependency isolation.

This script helps manage the version conflicts between Social-CADRL and other methods
by setting up a separate virtual environment with CADRL-specific dependencies.
"""

import os
import sys
import subprocess
import venv
from pathlib import Path

def create_cadrl_env():
    """Create a virtual environment specifically for CADRL."""
    cadrl_dir = Path("Methods/Social-CADRL")
    venv_dir = cadrl_dir / "venv"
    
    print("Setting up Social-CADRL environment...")
    print(f"Creating virtual environment at: {venv_dir}")
    
    # Remove existing environment if it exists
    if venv_dir.exists():
        print("Removing existing environment...")
        import shutil
        shutil.rmtree(venv_dir)
    
    # Create new virtual environment
    venv.create(venv_dir, with_pip=True)
    
    # Get python executable for the new environment
    if sys.platform == "win32":
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
    
    print("Installing CADRL-specific dependencies...")
    
    # Install CADRL-compatible versions
    dependencies = [
        "numpy==1.19.5",  # Compatible with numpy.bool8
        "tensorflow==2.8.0",  # Stable TensorFlow version
        "gym==0.21.0",  # Compatible gym version
        "matplotlib==3.5.3",
        "scipy==1.7.3",
        "opencv-python==4.5.5.64",
        "pandas==1.3.5"
    ]
    
    # Install dependencies one by one
    for dep in dependencies:
        print(f"Installing {dep}...")
        result = subprocess.run([str(pip_exe), "install", dep], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Failed to install {dep}")
            print(result.stderr)
    
    # Try to install gym-collision-avoidance
    print("Installing gym-collision-avoidance...")
    result = subprocess.run([str(pip_exe), "install", "gym-collision-avoidance"], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print("gym-collision-avoidance not available via pip, trying alternative...")
        # Install from source or use alternative method
        print("You may need to install gym-collision-avoidance manually")
        print("See: https://github.com/mit-acl/gym-collision-avoidance")
    
    print("✓ CADRL environment setup complete!")
    print(f"Environment location: {venv_dir}")
    print("\nTo activate manually:")
    if sys.platform == "win32":
        print(f"  {venv_dir}\\Scripts\\activate")
    else:
        print(f"  source {venv_dir}/bin/activate")

def check_cadrl_compatibility():
    """Check if current environment is compatible with CADRL."""
    print("Checking current environment compatibility...")
    
    try:
        import numpy as np
        numpy_version = np.__version__
        print(f"NumPy version: {numpy_version}")
        
        # Check for numpy.bool8 issue
        if hasattr(np, 'bool8'):
            print("✓ NumPy has bool8 attribute (compatible)")
        else:
            print("✗ NumPy missing bool8 attribute (needs older version)")
            return False
            
    except ImportError:
        print("✗ NumPy not installed")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow not installed")
        return False
    
    try:
        import gym
        print(f"✓ Gym version: {gym.__version__}")
    except ImportError:
        print("✗ Gym not installed")
        return False
    
    try:
        import gym_collision_avoidance
        print("✓ gym-collision-avoidance available")
    except ImportError:
        print("✗ gym-collision-avoidance not available")
        return False
    
    return True

def main():
    print("Social-CADRL Environment Setup")
    print("=" * 40)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Just check compatibility
        compatible = check_cadrl_compatibility()
        if compatible:
            print("\n✓ Current environment is compatible with CADRL")
        else:
            print("\n✗ Current environment has compatibility issues")
            print("Consider running this script without --check to create a dedicated environment")
        return
    
    print("\nThis script will create a virtual environment specifically for Social-CADRL")
    print("to avoid dependency conflicts with other methods.")
    print("\nOptions:")
    print("1. Create CADRL-specific environment (recommended)")
    print("2. Check current environment compatibility")
    print("3. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter choice (1-3): "))
            if choice in [1, 2, 3]:
                break
            print("Invalid choice! Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input! Please enter a number.")
    
    if choice == 1:
        create_cadrl_env()
    elif choice == 2:
        compatible = check_cadrl_compatibility()
        if compatible:
            print("\n✓ Current environment is compatible with CADRL")
        else:
            print("\n✗ Current environment has compatibility issues")
            print("Run option 1 to create a dedicated environment")
    else:
        print("Exiting...")

if __name__ == "__main__":
    main() 