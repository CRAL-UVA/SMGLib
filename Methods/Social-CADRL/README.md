# Social-CADRL
Continuous Action-Value Distributional Reinforcement Learning (CADRL) stands as a pivotal component within our repository, specializing in the realm of social navigation. It is a cutting-edge framework that harnesses the power of reinforcement learning to enable autonomous agents to navigate complex social environments seamlessly. CADRL leverages continuous action-value distributional reinforcement learning techniques to imbue agents with the ability to make dynamic decisions while considering the presence and behaviors of other agents. 

To run CADRL experiments for Social mini games go to specified folder and choose the scenario to run. For example to simulate doorway scenario with CADRL use following command

``` 
cd ./Methods/Social-CADRL/experiments/src 
python doorway.py 
```

The animation file for the simulation will be saved at 
``` ./Methods/Social-CADRL/experiments/results/example/animations ```

To simulate the nominal conditions for the agent in the given social scenario replace the test case file in the scenario file to 
``` tc.get_scenario_nominal() ```
for example to simulate nominal case for  the doorway scenario replace with
``` tc.get_doorway_nominal() ```

With CADRL, our repository equips agents to make informed, real-time decisions in congested scenarios, such as doorways, hallways, and intersections, ensuring not only efficient movement but also social adherence. CADRL plays a crucial role in enhancing the overall safety and functionality of multi-agent systems in social navigation contexts, making it an indispensable tool within our collection of algorithms and methodologies.

In the app options Agent have arguments
- `start_x` — (float or int) x position of agent in global frame at start of episode
- `start_y` — (float or int) y position of agent in global frame at start of episode
- `goal_x` — (float or int) desired x position of agent in global frame by end of episode
- ` goal_y` —(float or int) desired y position of agent in global frame by end of episoded
- `radius` — (float or int) radius of circle describing disc-shaped agent's boundaries in meters
- `pref_speed` —  (float or int) maximum speed of agent in m/s
- ` initial_heading` — (float) angle of agent in global frame at start of episode
- `policy` —  (:class:`~gym_collision_avoidance.envs.policies.Policy.Policy`) computes agent's action from its state
- `dynamics_model` —  (:class:`~gym_collision_avoidance.envs.dynamics.Dynamics.Dynamics`) computes agent's new state from its state and action
- `sensors` —  (list) of :class:`~gym_collision_avoidance.envs.sensors.Sensor.Sensor` measures the environment for use by the policy
- `id` —  (int) not sure how much it's used, but uniquely identifies each agent


# Setting up New Environment

For setting new environment go to 
``` ./envs/test_cases.py ```

Create a new test case with the Agent arguments accoridng to the need see 
``` config.json ``` for reference, then replace the test case file in line 34 at ./experiments/src/example.py
To see animation run the experiment file

``` python experiment.py ```

  
