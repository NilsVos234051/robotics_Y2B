import matplotlib.pyplot as plt
import numpy as np

# Add imports for the environment wrapper
from task10 import OT2Env

class PIDController:
    def __init__(self, kp, ki, kd, setpoint, output_limits=(-1.0, 1.0), time_step=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits  # Tuple (min, max)
        self.time_step = time_step

        self.previous_error = 0
        self.integral = 0

    def compute(self, current_position):
        # Calculate error
        error = self.setpoint - current_position

        # Proportional term
        p_term = self.kp * error

        # Integral term (scaled by time step)
        self.integral += error * self.time_step
        i_term = self.ki * self.integral

        # Derivative term (scaled by time step)
        derivative = (error - self.previous_error) / self.time_step
        d_term = self.kd * derivative

        # Compute raw output
        output = p_term + i_term + d_term

        # Clip output to limits
        output = max(self.output_limits[0], min(output, self.output_limits[1]))

        # Update previous error
        self.previous_error = error

        return output

# Example setup for PID controllers for each axis
def test_pid_with_simulation():
    # Initialize the simulation environment
    env = OT2Env(render=True, max_steps=1000)

    # Set up PID controllers for each axis
    kp, ki, kd = 40.0, 0.0, 0.00
    pid_x = PIDController(kp, ki, kd, setpoint=0)
    pid_y = PIDController(kp, ki, kd, setpoint=0)
    pid_z = PIDController(kp, ki, kd, setpoint=0)

    # Reset the environment
    observation, _ = env.reset()
    current_position = observation[:3]  # Current pipette position (X, Y, Z)

    # Prepare to collect data
    positions = []
    setpoints = [0.2, 0.2, 0.2]

    pid_x.setpoint = setpoints[0]
    pid_y.setpoint = setpoints[1]
    pid_z.setpoint = setpoints[2]

    for step in range(150):  # Run for 100 steps
        # Compute control actions
        action_x = pid_x.compute(current_position[0])
        action_y = pid_y.compute(current_position[1])
        action_z = pid_z.compute(current_position[2])

        # Combine into a single action (drop action is 0 for now)
        action = np.array([action_x, action_y, action_z, 0], dtype=np.float32)

        # Step the environment
        observation, reward, terminated, truncated, _ = env.step(action)

        # Update the current position
        current_position = observation[:3]
        positions.append(current_position)

        # Check termination
        if terminated or truncated:
            print("Simulation finished.")
            break

    # Convert collected data to a numpy array for plotting
    positions = np.array(positions)

    # Plot results
    plt.figure(figsize=(10, 6))
    print(np.sum(setpoints))
    plt.plot(np.sum(positions, axis=1), label="Position")
    plt.axhline(np.sum(setpoints), color="r", linestyle="--", label="Setpoint")
    plt.xlabel("Steps")
    plt.ylabel("Position")
    plt.title("PID Controller Convergence")
    plt.suptitle(F"P:{kp}, I:{ki}, D:{kd}")
    plt.legend()
    plt.grid()
    plt.show()

    # Close the environment
    env.close()

if __name__ == "__main__":
    test_pid_with_simulation()
