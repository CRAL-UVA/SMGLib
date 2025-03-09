# Social-QP-CBF

Control Barrier Functions methods contains two methods which are:
1.  Random CBF
2.  Game-Theortic CBF (GT-CBF)

CBFs methods are implemented in MATLAB. The code can be run with MATLAB 2014b and higher with MATLAB's optimization toolbox function 'quadprog.'
To simulate GT-CBF first go to specified folder and initalize the addition of directories
```0 
cd ./Methods/Social-QP-CBF/
init.m
```

## Random CBF
Random CBF (Control Barrier Functions) for Social Navigation: Within the context of social navigation, Random CBF presents an innovative approach that employs controlled random perturbations to ensure smooth agent interactions while avoiding deadlocks. Agents operating within shared spaces like doorways, hallways, and intersections benefit from this method, as it enhances safety and efficiency through a unique strategy. Random CBF relies on safety barrier certificates and Control Barrier Functions, which act as safeguards against collisions and deadlock scenarios. These functions are designed to ensure that agents maintain safe distances and avoid unwanted interactions. The key innovation lies in introducing controlled randomness into agent movements, effectively breaking potential deadlocks and enabling dynamic decision-making.

To simulate Random CBF for social navigation scenario use the scenario file within the options below:
1. Random_QP_CBF_Doorway.m
2. Random_QP_CBF_Intersection.m
3. Random_QP_CBF_Hallway.m
   
For example to run hallway scenario with random perturbation

``` Random_QP_CBF_Hallway.m ```

By introducing small, calculated perturbations to agent trajectories, Random CBF promotes the redistribution of agents within confined spaces. This dynamic approach prevents agents from getting stuck in gridlock situations and encourages them to explore alternative routes. While maintaining safety, Random CBF introduces an element of adaptability, making it an asset in scenarios where unpredictable agent behaviors and congestion prevail. Random CBF's integration into our repository underscores our commitment to exploring diverse solutions for social navigation. Its ability to combine safety, adaptability, and efficiency positions it as a valuable tool for orchestrating the seamless movement of agents in complex social environments.

## GT-CBF
Game-Theoretic CBF (Cooperative Behavior Formation) for Social Navigation: In the realm of social navigation, Game-Theoretic CBF emerges as a strategic approach that leverages the principles of game theory to foster cooperative and safe interactions among agents. Each agent within a multi-agent system adheres to a defined game-theoretic relation, where the maximum velocity of an agent is intelligently regulated. This methodology fosters a cooperative atmosphere among agents, promoting harmonious movement within shared spaces such as hallways, doorways, and intersections. By dynamically adjusting velocities based on the number of agents present, Game-Theoretic CBF mitigates congestion and minimizes the likelihood of collisions. It optimizes the utilization of available space, ensuring that each agent moves efficiently while respecting the presence and rights of others.

To run Game Therotic CBF for social navigation scenario first comment line 132-142 and uncomment line 148-158 in ARobotarium.m. Run the file with sepcified scenarion from below options to simulate 
1. GT_QP_CBF_Doorway.m
2. GT_QP_CBF_Intersection.m
3. GT_QP_CBF_Hallway.m

For example to run doorway scenario wirth Game Theoritc CBF run

```  GT_QP_CBF_Doorway.m ```

The animation will be saved at ./examples/

Game-Theoretic CBF serves as a pivotal component within our repository, contributing to the development of intelligent and socially aware multi-agent systems. Its application ensures that agents navigate through complex social environments in a manner that prioritizes safety and fluidity, making it an indispensable asset for social navigation scenarios.


# Setting up New Environment

To create and test new environment change the configurations form the file you want to change.
- `N` - Number of Agents
- `x` - Position of each agent
- `goal_condition` - Goal positions for agents
- `writerObj`-  Name of saved video

run the file to test the scenario with new envirionment
