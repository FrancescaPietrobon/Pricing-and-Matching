from Simulator import *
from LP_optimization import *

simulator = Simulator()
simulator.simulation_step_1()

# Creating the inputs to give to the LP_optimization

# Price
price = simulator.get_item2().get_price()

# Promo codes
p1 = 0.1
p2 = 0.2
p3 = 0.5

# Conversion rates
pr_c1_p0 = simulator.group1.get_conversion_rate_item2_given_item1_P0()
pr_c2_p0 = simulator.group2.get_conversion_rate_item2_given_item1_P0()
pr_c3_p0 = simulator.group3.get_conversion_rate_item2_given_item1_P0()
pr_c4_p0 = simulator.group4.get_conversion_rate_item2_given_item1_P0()

pr_c1_p1 = simulator.group1.get_conversion_rate_item2_given_item1_P1()
pr_c2_p1 = simulator.group2.get_conversion_rate_item2_given_item1_P1()
pr_c3_p1 = simulator.group3.get_conversion_rate_item2_given_item1_P1()
pr_c4_p1 = simulator.group4.get_conversion_rate_item2_given_item1_P1()

pr_c1_p2 = simulator.group1.get_conversion_rate_item2_given_item1_P2()
pr_c2_p2 = simulator.group2.get_conversion_rate_item2_given_item1_P2()
pr_c3_p2 = simulator.group3.get_conversion_rate_item2_given_item1_P2()
pr_c4_p2 = simulator.group4.get_conversion_rate_item2_given_item1_P2()

pr_c1_p3 = simulator.group1.get_conversion_rate_item2_given_item1_P3()
pr_c2_p3 = simulator.group2.get_conversion_rate_item2_given_item1_P3()
pr_c3_p3 = simulator.group3.get_conversion_rate_item2_given_item1_P3()
pr_c4_p3 = simulator.group4.get_conversion_rate_item2_given_item1_P3()

# Maximum number of promo codes
max_p0 = simulator.get_p0_num()
max_p1 = simulator.get_p1_num()
max_p2 = simulator.get_p2_num()
max_p3 = simulator.get_p3_num()

# Maximum number of customers per class
max_n1 = simulator.get_day_step1().get_number_of_customers_c1()
max_n2 = simulator.get_day_step1().get_number_of_customers_c2()
max_n3 = simulator.get_day_step1().get_number_of_customers_c3()
max_n4 = simulator.get_day_step1().get_number_of_customers_c4()

# Calling the optimization function
fun, result = LP(price, p1, p2, p3,
                 pr_c1_p0, pr_c2_p0, pr_c3_p0, pr_c4_p0,
                 pr_c1_p1, pr_c2_p1, pr_c3_p1, pr_c4_p1,
                 pr_c1_p2, pr_c2_p2, pr_c3_p2, pr_c4_p2,
                 pr_c1_p3, pr_c2_p3, pr_c3_p3, pr_c4_p3,
                 max_p0, max_p1, max_p2, max_p3,
                 max_n1, max_n2, max_n3, max_n4)

print(fun)
print("\n")
print(result)
