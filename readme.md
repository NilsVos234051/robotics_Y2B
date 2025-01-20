# README.md

## Robotic Simulation for Opentron OT-2 Pipette Working Envelope

This repository contains the code to determine the working envelope of the Opentron OT-2 robot's pipette in a simulated environment. The simulation adjusts motor velocities for each axis to move the pipette to the corners of the cube defining its working envelope.

---

### Requirements

Ensure the following dependencies are installed:

- Python 3.8+
- `numpy`
- `Pillow`
- `matplotlib` (optional, for visualizations)
- Simulation environment (refer to setup instructions)

Install dependencies using pip:

```bash
pip install numpy pillow matplotlib
```

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/BredaUniversityADSAI/2024-25b-fai2-adsai-NilsVos234051
   cd 2024-25b-fai2-adsai-NilsVos234051/robotics_tasks
   ```

2. Ensure you have the simulation environment installed and configured. Follow the instructions provided in the simulation documentation.

3. Ensure the `sim_class.py` file is in the same directory as the script.

### Running the Simulation

To run the script and determine the working envelope, execute:

```bash
python task9.py
```

The script will:
- Simulate the robot moving to each corner of its working envelope.
- Print the coordinates of the corners and the X, Y, Z ranges.

---

### Code Explanation

The script:
1. Initializes a simulation environment with one agent.
2. Iteratively adjusts motor velocities for each axis to find all 8 corners of the working envelope.
3. Prints the working envelope coordinates and saves a GIF of the simulation.

#### Main Code File: `task9.py`

```python
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
```

---

### Results

The working envelope coordinates of the pipette will be printed as follows:

- **Corners:** A list of all 8 corner coordinates.
- **Ranges:**
  - X Range: (-0.1873, 0.2534)
  - Y Range: (-0.1706, 0.2197)
  - Z Range: (0.1692, 0.2897)
