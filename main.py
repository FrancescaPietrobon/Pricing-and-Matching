from Simulator import *
from LP_optimization import *

simulator_1 = Simulator(365)
simulator_2 = Simulator(365)

# Step 1
# Experiment 1
experiment_1 = simulator_1.simulation_step_1(0.7, 0.2, 0.07, 0.03)
print(experiment_1)

# Experiment 2
experiment_2 = simulator_2.simulation_step_1(0.6, 0.25, 0.1, 0.05)
print(experiment_2)
