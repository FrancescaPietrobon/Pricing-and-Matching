from Simulator import *
import pandas as pd

p_frac_exp1 = np.array([0.7, 0.2, 0.07, 0.03])
p_frac_exp2 = np.array([0.6, 0.25, 0.1, 0.05])

print("\t\tData Intelligence Applications - A.Y. 2020/2021")
print("\t\t\t\tPricing and Matching project\n")

while True:
    choice = int(input("Select the step (1-3-4-5-6-7-8): "))

    if choice == 1:
        experiment_1_1 = Simulator().simulation_step_1(p_frac_exp1)
        result = pd.DataFrame(data=experiment_1_1[1],
                              index=["P0", "P1", "P2", "P3"],
                              columns=["Class1", "Class2", "Class3", "Class4"])
        print("\nExperiment 1: fractions", p_frac_exp1, "\nOptimal assignment of promos to customer classes:\n", result)

        experiment_1_2 = Simulator().simulation_step_1(p_frac_exp2)
        result = pd.DataFrame(data=experiment_1_2[1],
                              index=["P0", "P1", "P2", "P3"],
                              columns=["Class1", "Class2", "Class3", "Class4"])
        print("\nExperiment 2: fractions", p_frac_exp2, "\nOptimal assignment of promos to customer classes:\n", result)
        break

    elif choice == 3:
        step3 = Simulator().simulation_step_3()
        break

    elif choice == 4:
        step4 = Simulator().simulation_step_4()
        break

    elif choice == 5:
        step5 = Simulator().simulation_step_5(0.2, 0.07, 0.03)
        break

    elif choice == 6:
        step6 = Simulator().simulation_step_6(0.7, 0.2, 0.07, 0.03)
        break

    elif choice == 7:
        step7 = Simulator().simulation_step_7(0.7, 0.2, 0.07, 0.03)
        break

    elif choice == 8:
        print("This step has not been implemented yet.")
        #step7 = Simulator().simulation_step_8(0.7, 0.2, 0.07, 0.03)
        break

    else:
        print("You entered an invalid step number. Please try again.")
