from Simulator import *

simulator = Simulator()
simulator.simulation_step_1()

# Creating the inputs to give to the weighted_constrained_matching_algorithm
# Vector c = number of customers [c1, c2, c3, c4]
num_customers = [simulator.get_day_step1().get_number_of_customers_c1(),
                 simulator.get_day_step1().get_number_of_customers_c2(),
                 simulator.get_day_step1().get_number_of_customers_c3(),
                 simulator.get_day_step1().get_number_of_customers_c4()]

# Matrix v = constraints row promo / column classes (without P0 as it is infinity)

