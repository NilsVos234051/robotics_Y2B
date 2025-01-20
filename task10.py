import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(
            low=np.array([-0.1873, -0.1706, 0.1692, 0], dtype=np.float32),  # Add 0 as the minimum for drop action
            high=np.array([0.2534, 0.2197, 0.2897, 1], dtype=np.float32),  # Add 1 as the maximum for drop action
            shape=(4,),  # 4 values in the action space (3 for movement, 1 for drop action)
            dtype=np.float32
        )
        # Observation consists of the current pipette position (3 values: x, y, z), and the goal position (3 values).
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
                                            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                                            dtype=np.float32)

        # keep track of the number of steps
        self.steps = 0

    def reset(self, seed=None):
        # Being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # Set a random goal position for the agent, consisting of x, y, and z coordinates within the working area
        self.goal_position = np.random.uniform(low=-1.0, high=1.0, size=3)

        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)

        # Dynamically find the robot ID key
        robot_key = next(iter(observation.keys()))

        # Now we need to process the observation and extract the relevant information, the pipette position,
        # convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        pipette_position = np.array(observation[robot_key]['pipette_position'], dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Reset the number of steps
        self.steps = 0

        info = {}

        return observation, info

    def step(self, action):
        # Execute one time step within the environment
        # Since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = np.append(action, 0)

        # Call the environment step function
        observation = self.sim.run([action])

        # Now we need to process the observation and extract the relevant information, the pipette position,
        # convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        robot_key = next(iter(observation.keys()))
        pipette_position = np.array(observation[robot_key]['pipette_position'], dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position], dtype=np.float32)

        # Calculate the reward
        # Example reward: negative distance from goal, higher reward when closer to goal
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        reward = -distance_to_goal  # Simple reward, you may adjust this

        # Check if the task is complete (if distance is below a threshold)
        if distance_to_goal < 0.05:  # Example threshold, adjust based on the task size
            terminated = True
            reward += 10  # Reward for completing the task
        else:
            terminated = False

        # Check if the episode should be truncated (if the number of steps exceeds max_steps)
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {}  # We don't need to return any additional information

        # Increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()
