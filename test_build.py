#!/usr/bin/env python3
"""
Test script to verify Social-ORCA build process.
This script tests the build_social_orca function to ensure it works correctly.
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import from run_simulation
sys.path.insert(0, str(Path(__file__).parent))

from run_simulation import build_social_orca

def test_build():
    """Test the Social-ORCA build process."""
    print("Testing Social-ORCA build process...")
    print("=" * 50)
    
    # Test the build function
    success = build_social_orca()
    
    if success:
        print("\n✅ BUILD TEST PASSED")
        print("The Social-ORCA executable was successfully built.")
        return True
    else:
        print("\n❌ BUILD TEST FAILED")
        print("The Social-ORCA executable could not be built.")
        return False

if __name__ == "__main__":
    success = test_build()
    sys.exit(0 if success else 1) 