from Simulator import *
from LP_optimization import *


# Step 1
simulator_step1_1 = Simulator(365)
simulator_step1_2 = Simulator(365)

# Experiment 1
experiment_1_1 = simulator_step1_1.simulation_step_1(0.7, 0.2, 0.07, 0.03)
print(experiment_1_1)

# Experiment 2
experiment_1_2 = simulator_step1_2.simulation_step_1(0.6, 0.25, 0.1, 0.05)
print(experiment_1_2)

'''
# Step 2
simulator_step2_1 = Simulator(365)
simulator_step2_2 = Simulator(365)

# Experiment 1
experiment_2_1 = simulator_step2_1.simulation_step_2(0.7, 0.2, 0.07, 0.03)

# Experiment 2
experiment_2_2 = simulator_step2_2.simulation_step_2(0.6, 0.25, 0.1, 0.05)
'''