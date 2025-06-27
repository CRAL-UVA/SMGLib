#!/bin/bash

# Make the Python script executable
chmod +x generate_config.py

# Run the configuration generator
python3 generate_config.py

# Check if the configuration file was generated
if [ $? -eq 0 ]; then
    # Get the most recently created config file
    CONFIG_FILE=$(ls -t config_*.xml | head -n1)
    
    # Build the project if needed
    if [ ! -d "build" ]; then
        mkdir build
        cd build
        cmake ..
        make
        cd ..
    fi
    
    # Run the simulation with the generated config file
    echo "Running simulation with configuration: $CONFIG_FILE"
    ./build/Social-ORCA $CONFIG_FILE
else
    echo "Failed to generate configuration file"
    exit 1
fi 