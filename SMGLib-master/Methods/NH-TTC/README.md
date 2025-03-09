# NH-TTC: A generalized framework for anticipatory collision avoidance
This repository hosts the companion code to the RSS 2020 paper "NH-TTC: A generalized framework for anticipatory collision avoidance".
Within this repository is both the code for the collision avoidnace, as well as a live, graphical frontend.

Please see the corresponding webpage for videos of results: http://motion.cs.umn.edu/r/NH-TTC/

## Contents
* [Building](#building)
  * [Linux](#linux)
  * [Windows](#windows)
  * [OSX](#osx)
* [Live Demo Usage](#live-demo-usage)

## Building
### Linux
Install all required components.  On a fresh Ubuntu install:
```
apt install git cmake g++ xorg-dev libgtk-3-dev
```

Clone the repository and all submodules:
```
git clone https://github.com/davisbo/NHTTC.git
cd NHTTC
git submodule init
git submodule update
```

Build the repository
```
mkdir build
cd build
cmake ..
make
```

Run the demo
```
cd src/nhttc_opengl
./run_nhttc
```

### Windows
The windows build is set up for using CMake (https://cmake.org/) and Visual Studio (https://visualstudio.microsoft.com/)
Once these are installed, clone the repository and all submodules:
```
git clone https://github.com/davisbo/NHTTC.git
cd NHTTC
git submodule init
git submodule update
```

Configure and generate the project file with CMake, and then open the project file in Visual Studio.
The code can now be compiled and run as normal.

### OSX
Ensure you have git, cmake, and the xcode dev tools installed.

Clone the repository and all submodules:
```
git clone https://github.com/davisbo/NHTTC.git
cd NHTTC
git submodule init
git submodule update
```

Build the repository
```
mkdir build
cd build
cmake ..
make
```

Run the demo
```
cd src/nhttc_opengl
./run_nhttc
```

## Live Demo Usage
* Click and drag to move the camera.  Scroll the mouse to zoom in or out.
* 'Space': Pauses or Resumes execution.
* 'C': Clears all agents from the simulation.
* 'L': Brings up a file dialog to load a scene file. The 'scenes' folder contains a variety of scenes used in the corresponding paper.
* 'S': Brings up a file dialog to save the current agent positions and goal positions to a scene file. Note that this clears the current scene.
* 'R': Swaps the initial position of an agent and its goal, giving a quick way to reverse a simulation.
* 'A': Enters adding mode
  * First, use the scroll wheel or arrow keys to select an agent type.  In order they are: velocity, acceleration, differential drive, smooth differential drive, simple car, smooth car, and then non-reactive versions of each (i.e. agents that don't account for any other agents in planning). Click to lock in that agent type.
  * Next, for any agent with orientation, move the mouse to orient the agent.  Click once the desired orientation is reached.
  * Finally, move the mouse to the desired goal location.  Click to lock this in.
  * Note that entering adding mode automatically pauses the simulation.
