# Social ORCA-MAPF
ORCA-MAPF (Optimal Reciprocal Collision Avoidance for Multi-Agent Path Finding): Within our repository, ORCA-MAPF takes center stage as a robust and efficient solution designed to address multi-agent pathfinding challenges in dynamic social environments. This state-of-the-art algorithm leverages the principles of optimal reciprocal collision avoidance to enable autonomous agents to navigate complex scenarios, such as crowded doorways, hallways, and intersections, with precision and safety. ORCA-MAPF excels in optimizing the trajectories of multiple agents simultaneously, considering not only their paths but also their velocities and intentions.

To run the ORCA-MAPF for social mini game scenario first build the project.

```bash
cd ./Methods/Social-ORCA/
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE={Release/Debug} -DFULL_OUTPUT_FLAG={ON/OFF} -DFULL_LOG_FLAG={ON/OFF} -DMAPF_LOG_FLAG={ON/OFF} ..
make
```

where optons:
- `CMAKE_BUILD_TYPE` — Standard CMake option that specifies the build type. For more information see [[CMake Documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)]    
etc
  - `Release` uses to build with no debugging information. 
  - `Debug` usually uses to enable debugging information, disable optimization 
- `FULL_OUTPUT_FLAG` — Enables/disables full output to stdout about reading xml files.
- `FULL_LOG_FLAG` — Enables/disables logging to xml file agents state information
- `MAPF_LOG_FLAG` — Enables/disables logging to xml files information about MAPF instances in coordinated mode

After this launch the social navigation scenario to run using following command
```bash
./single_test {file_name num}
```
where 
- `file_name` — name of the XML file, which contain social; scenario;
- `num` — the number of agents with which tasks will run.

For example to simulate intersection scenario use following command

```bash
./single_test ../task_examples/intersection.xml 2
```
Summary will be displayed after execution using standard output, full log (if such option in in CMake was chosen) 
will be saved in same directory as task file and will be named according to the following pattern:
```
*taskfilename*_*number_of_agents*_log.xml
```
For example:
```
intersection_2_log.xml
```
By incorporating ORCA-MAPF into our collection, we enhance the capability of agents to make real-time decisions that promote both smooth movement and social compliance. It represents a key component in our endeavor to advance the science and technology of multi-agent systems for social navigation, contributing to safer and more efficient interactions between agents in shared spaces.

Input files are an XML files with a specific structure.  
Input file should contain:

* Mandatory tag `<agents>`. It describes the parameters of the agents.
    * `number` — mandatory attribute that define the number of agents;
    * `type` — attribute that define the type of agents. Possible values: 
      - `orca` — running the algorithm in standard mode without avoiding deadlocks
      - `orca-par` — running the algorithm with avoiding deadlocks using Push and Rotate algorithm to create common coordinated plan. For more detains see out paper "A Combination of Theta*, ORCA and Push and Rotate for Multi-agent Navigation".
      - `orca-par-ecbs` — running the algorithm with avoiding deadlocks using combination of Push and Rotate and ECBS algorithms to create common coordinated plan. For more detains see out paper "Distributed Multi-Agent Navigation Based on Reciprocal Collision Avoidance and Locally Confined Multi-Agent Path Finding".
      - `orca-return` — experimental mode in which random agents return to their previous waypoints in case of deadlocks.
    * `<default_parameters>` — mandatory tags that defines default parameters of agents and agent's perception.
      * `agentsmaxnum` — mandatory attribute that defines a number of neighbors, that the agent takes into account;
      * `movespeed` — mandatory attribute that defines maximum speed of agent;
      * `sightradius` — mandatory attribute that defines the radius in which the agent takes neighbors into account;
      * `size` — mandatory attribute that defines size of the agent (radius of the agent);
      * `timeboundary` — mandatory attribute that defines the time within which the algorithm ensures collision avoidance with other agents;
      * `timeboundaryobst` — mandatory attribute that defines the time within which the algorithm ensures collision avoidance with static obstacles.
    * `<agent>` — mandatory tags that defines parameters of each agent.
        * `id` — mandatory attribute that defines the identifier of agent;
        * `start.xr` — mandatory attribute that defines the coordinate of start position on the x-axis (hereinafter, excluding `map` tag, points (x,y) are in coordinate system, which has an origin (0,0) in lower left corner. More about coordinate systems in the illustration below);
        * `start.yr` — mandatory attribute that defines the coordinate of start position on the y-axis; 
        * `goal.xr` — mandatory attribute that defines the coordinate of finish position on the x-axis; 
        * `goal.yr` — mandatory attribute that defines the coordinate of finish position on the y-axis; 
        * `agentsmaxnum` — attribute that defines a number of neighbors, that the agent takes into account;
        * `movespeed` — attribute that defines maximum speed of agent;
        * `sightradius` — attribute that defines the radius in which the agent takes neighbors into account;
        * `size` — attribute that defines size of the agent (radius of the agent);
        * `timeboundary` — attribute that defines the time within which the algorithm ensures collision avoidance with other agents;
        * `timeboundaryobst` — attribute that define the time within which the algorithm ensures collision avoidance with static obstacles.
* Mandatory tag `<map>`. It describes the environment for global path planning.
  * `<height>` and `<width>` — mandatory tags that define size of the map. Origin is in the upper left corner. (0,0) - is upper left, (*width*-1, *height*-1) is lower right (more about coordinate systems in the illustration below). 
  * `<cellsize>` — optional tag that defines the size of one cell.
  * `<grid>` — mandatory tag that describes the square grid constituting the map. It consists of `<row>` tags. Each `<row>` contains a sequence of "0" and "1" separated by blanks. "0" stands for traversable cell, "1" — for untraversable (actually any other figure but "0" can be used instead of "1").

* Mandatory tag `<obstacles>`. It describes static obstacles for collision avoidance.
  * `number` — mandatory attribute that defines the number of obstacles;
  * `<obstacle>` — mandatory tags which defines each static obstacles for collision avoidance.
    * `<vertex>` — mandatory tags which defines vertex of static obstacle for collision avoidance. 
      *  `xr` — mandatory attribute that defines the coordinate of vertex on the x-axis; 
      *  `yr` — mandatory attribute that defines the coordinate of vertex on the y-axis.
  
* Mandatory tag `<algorithm>`. It describes the parameters of the algorithm.
  * `<delta>` — mandatory tag that defines the distance between the center of the agent and the finish, which is enough to reach the finish (ORCA parameter);
  * `<timestep>` — mandatory tag that defines the time step of simulation (ORCA parameter);
  * `<searchtype>` — tag that defines the type of planning. Possible values - "thetastar" (use Theta* for planning), "direct" (turn off global planning and always use direction to global goal). Default value is "thetastar" (global planning parameter);
  * `<breakingties>` — tag that defines the priority in OPEN list for nodes with equal f-values. Possible values - "0" (break ties in favor of the node with smaller g-value), "1" (break ties in favor of the node with greater g-value). Default value is "0" (Theta* parameter);
  * `<cutcorners>` — boolean tag that defines the possibility to make diagonal moves when one adjacent cell is untraversable. The tag is ignored if diagonal moves are not allowed. Default value is "false" (Theta* parameter);
  * `<allowsqueeze>` — boolean tag that defines the possibility to make diagonal moves when both adjacent cells are untraversable. The tag is ignored if cutting corners is not allowed. Default value is "false" (Theta* parameter);
  * `<hweight>` — defines the weight of the heuristic function. Should be real number greater or equal 1. Default value is "1" (Theta* parameter);
  * `<trigger>` — defines the switching to the coordinated mode trigger. Possible values:
    - `speed-buffer` — use average speed for deadlock detection. For more detains see out paper "Distributed Multi-Agent Navigation Based on Reciprocal Collision Avoidance and Locally Confined Multi-Agent Path Finding".
    - `common-point` — use information about movement to a common waypoint for deadlock detection. For more detains see out paper "A Combination of Theta*, ORCA and Push and Rotate for Multi-agent Navigation".
  * `<mapfnum>` — defines the minimum number of agents that should move towards a common waypoint (for `common-point` trigger)

### Other Options

In `src/experiments/single_test.cpp`
- `STEP_MAX` — the maximum number of simulation steps.
- `STOP_BY_SPEED` — enables/disables instance execution halting depending on the average speed of the agents.
- `IS_TIME_BOUNDED` — enables/disables instance execution halting depending on the runtime.
- `TIME_MAX` — the maximum runtime in ms.

# Setting up New Environment

To create and test new enviropnment create a new scxenario xml file in task_examples folder and set the needed values for flags given above. For reference see
``` config.json ```

To test the new enviornment with name new_scenario.xml in task_examples
```
./single_test ../task_examples/intersection.xml {num_agents}
```
