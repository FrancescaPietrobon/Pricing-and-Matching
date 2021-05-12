from Simulator import *

while True:
    choice = int(input("Step number: "))

    if choice == 1:
        # Step 1
        simulator_step1_1 = Simulator()
        simulator_step1_2 = Simulator()

        # Experiment 1
        experiment_1_1 = simulator_step1_1.simulation_step_1(0.7, 0.2, 0.07, 0.03)
        print(experiment_1_1)

        # Experiment 2
        experiment_1_2 = simulator_step1_2.simulation_step_1(0.6, 0.25, 0.1, 0.05)
        print(experiment_1_2)
        break

    elif choice == 3:
        # Step 3
        simulator_step3 = Simulator()
        step3 = simulator_step3.simulation_step_3()
        break

    elif choice == 4:
        # Step 4
        simulator_step4 = Simulator()
        step4 = simulator_step4.simulation_step_4()
        break

    elif choice == 5:
        # Step 5
        simulator_step5 = Simulator()
        step5 = simulator_step5.simulation_step_5(0.2, 0.07, 0.03)
        break

    elif choice == 6:
        # Step 6
        simulator_step6 = Simulator()
        step6 = simulator_step6.simulation_step_6(0.7, 0.2, 0.07, 0.03)
        break

    elif choice == 7:
        # Step 7
        simulator_step7 = Simulator()
        step7 = simulator_step7.simulation_step_7(0.7, 0.2, 0.07, 0.03)
        break

    else:
        print("You entered an invalid step number. Please try again.")
