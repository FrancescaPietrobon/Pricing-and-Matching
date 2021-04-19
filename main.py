from Simulator import *


# Step 1
simulator_step1_1 = Simulator()
simulator_step1_2 = Simulator()

# Experiment 1
experiment_1_1 = simulator_step1_1.simulation_step_1(0.7, 0.2, 0.07, 0.03)
print(experiment_1_1)

# Experiment 2
experiment_1_2 = simulator_step1_2.simulation_step_1(0.6, 0.25, 0.1, 0.05)
print(experiment_1_2)


# Step 3
simulator_step3 = Simulator()
step3 = simulator_step3.simulation_step_3()
