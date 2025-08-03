# Social-IMPC-DR
IMPC-DR for Social Navigation: IMPC-DR, or Infinite-Horizon Model Predictive Control with Deadlock Resolution, holds immense promise in the context of social navigation scenarios. In bustling environments characterized by doorways, hallways, and intersections, where multiple agents, be they pedestrians or autonomous robots, coexist, IMPC-DR shines as a beacon of efficiency and safety. By providing rigorous guarantees on optimization feasibility, IMPC-DR ensures collision-free trajectory generation for agents within shared workspaces. It not only tackles the challenge of resolving deadlocks but also preemptively detects and resolves potential deadlocks through its adaptive resolution scheme. This proactive approach promotes smooth agent movements and prevents gridlocks, even in the most crowded and high-speed scenarios.

To simulate IMPC-DR for social mini games go to specified folder and launch the app to run specific social scenario

``` 
cd ./Methods/Social-IMPC-DR/ 
python app.py
```
After the simulation is completed the animation will be saved at 
``` ./Methods/Social-IMPC-DR/ ```

In the app options
- `h` — Time Step
- `Max_Step` — Maximum number of time steps for simulation
- `Number` — Number of Agents
- `K` —  Length of Horizon
- `r_min` — minimum radius
- `epsilon` — width of warning band
- `start animation` —  to start the animation
- `save animation` —  to save the animation with name results01.avi
- `save trajectory` —  save the trajectory with name trajectory.svg
- `random original state` — randomly selects orihinal state for agents
- `random target` — randomly selects target state for agents
  
IMPC-DR is a key asset within our repository, contributing to the enhancement of multi-agent systems' performance in social navigation contexts. Its ability to combine theoretical rigor with practical effectiveness makes it an invaluable tool for creating safe and efficient interactions between agents navigating shared spaces.

# Setting up New Environment

To simulate a new expeiment use the argument with needed value, for reference see 
``` config.json ```
Use ``` Set original State ``` and ``` Set target ``` and the use ``` start animation ``` to test the new envrionment
