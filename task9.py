from sim_class import Simulation
import numpy as np

# Initialize the simulation with a specified number of agents
sim = Simulation(num_agents=1)

velocity_x = 2
velocity_y = 0
velocity_z = 2
drop_command = 0

actions = [[velocity_x, velocity_y, velocity_z, drop_command],
           [velocity_x, velocity_y, velocity_z, drop_command]]

state = sim.run(actions)

changes = 0
z = 0

corners = []

# Run the simulation for a specified number of steps
for i in range(1000):
    robotId_1_pipette_old = state['robotId_1']['pipette_position']

    actions = [[velocity_x, velocity_y, velocity_z, drop_command],
               [velocity_x, velocity_y, velocity_z, drop_command]]

    state = sim.run(actions)

    # Extract pipette positions
    robotId_1_pipette = state['robotId_1']['pipette_position']
    if robotId_1_pipette == robotId_1_pipette_old:
        if changes == 0:
            corners.append(state['robotId_1']['pipette_position'])
            velocity_x = 0
            velocity_y = 1
            changes += 1
        elif changes == 1:
            corners.append(state['robotId_1']['pipette_position'])
            velocity_x = -1
            velocity_y = 0
            changes += 1
        elif changes == 2:
            corners.append(state['robotId_1']['pipette_position'])
            velocity_x = 0
            velocity_y = -1
            changes += 1
        elif changes == 3:
            corners.append(state['robotId_1']['pipette_position'])
            velocity_x = 1
            velocity_y = 0
            changes += 1
        elif changes == 4 and z == 0:
            velocity_x = 0
            velocity_y = 0
            velocity_z = -1
            changes = 0
            z = 1
        elif changes == 4 and z == 1:
            corners.append(state['robotId_1']['pipette_position'])
            # Log the corners in the terminal.
            print("Found all corners.")
            print("Corners:", corners)

            corners = np.asarray(corners)

            # Calculate min and max for each axis
            x_min, y_min, z_min = corners.min(axis=0)
            x_max, y_max, z_max = corners.max(axis=0)

            print("X Range:", (x_min, x_max))
            print("Y Range:", (y_min, y_max))
            print("Z Range:", (z_min, z_max))
            break

sim.close()