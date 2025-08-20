#!/usr/bin/env python3
"""
SMGLib Installation Verification Script

This script verifies that all components of SMGLib are properly installed and working.
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 6:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.6+")
        return False

def check_dependencies():
    """Check if required Python packages are installed."""
    print("\nChecking Python dependencies...")
    required_packages = [
        'numpy', 'matplotlib', 'pandas', 'scipy', 'xml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True

def check_build_tools():
    """Check if required build tools are available."""
    print("\nChecking build tools...")
    
    tools = ['g++', 'make']
    missing_tools = []
    
    for tool in tools:
        try:
            result = subprocess.run(['which', tool], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {tool} - Found at {result.stdout.strip()}")
            else:
                print(f"❌ {tool} - Not found")
                missing_tools.append(tool)
        except Exception:
            print(f"❌ {tool} - Error checking")
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"\nMissing build tools: {', '.join(missing_tools)}")
        print("Install build-essential package (Ubuntu/Debian) or equivalent")
        return False
    return True

def check_social_orca_build():
    """Check if Social-ORCA can be built."""
    print("\nChecking Social-ORCA build...")
    
    try:
        # Import the build function
        sys.path.insert(0, str(Path(__file__).parent))
        from run_simulation import build_social_orca
        
        success = build_social_orca()
        if success:
            print("✅ Social-ORCA build - Successful")
            return True
        else:
            print("❌ Social-ORCA build - Failed")
            return False
    except Exception as e:
        print(f"❌ Social-ORCA build - Error: {e}")
        return False

def check_directory_structure():
    """Check if the expected directory structure exists."""
    print("\nChecking directory structure...")
    
    base_dir = Path(__file__).parent
    required_dirs = [
        'src/methods/Social-ORCA',
        'src/methods/Social-IMPC-DR', 
        'src/methods/Social-CADRL',
        'logs'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"✅ {dir_path} - Exists")
        else:
            print(f"❌ {dir_path} - Missing")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\nMissing directories: {', '.join(missing_dirs)}")
        return False
    return True

def main():
    """Run all verification checks."""
    print("SMGLib Installation Verification")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directory_structure),
        ("Build Tools", check_build_tools),
        ("Python Dependencies", check_dependencies),
        ("Social-ORCA Build", check_social_orca_build),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} - Error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("VERIFICATION SUMMARY")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! SMGLib is ready to use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 