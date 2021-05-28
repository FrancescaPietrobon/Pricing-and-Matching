from Simulator import *
import pandas as pd


def main():
    p_frac_exp1 = np.array([0.7, 0.2, 0.07, 0.03])
    p_frac_exp2 = np.array([0.6, 0.25, 0.1, 0.05])

    print("\t\tData Intelligence Applications - A.Y. 2020/2021")
    print("\t\t\t\tPricing and Matching project\n")

    while True:
        choice = int(input("Select the step (1-3-4-5-6-7-8): "))

        if choice == 1:
            revenue, best_price_item1, best_price_item2, matching = Simulator().simulation_step_1(p_frac_exp1)
            result = pd.DataFrame(data=matching,
                                  index=["P0", "P1", "P2", "P3"],
                                  columns=["Class1", "Class2", "Class3", "Class4"])
            print("\nExperiment 1: fractions", p_frac_exp1,
                  "\nRevenue:", revenue,
                  "\nOptimal price item 1:", best_price_item1,
                  "\nOptimal price item 2:", best_price_item2,
                  "\nOptimal assignment of promos to customer classes:\n", result)

            revenue, best_price_item1, best_price_item2, matching = Simulator().simulation_step_1(p_frac_exp2)
            result = pd.DataFrame(data=matching,
                                  index=["P0", "P1", "P2", "P3"],
                                  columns=["Class1", "Class2", "Class3", "Class4"])
            print("\nExperiment 2: fractions", p_frac_exp2,
                  "\nRevenue:", revenue,
                  "\nOptimal price item 1:", best_price_item1,
                  "\nOptimal price item 2:", best_price_item2,
                  "\nOptimal assignment of promos to customer classes:\n", result)
            break

        elif choice == 3:
            experiment_3_1 = Simulator().simulation_step_3(p_frac_exp1)
            plot_regret("STEP 3 - EXP 1", experiment_3_1[0], experiment_3_1[1], experiment_3_1[2])
            plot_reward("STEP 3 - EXP 1", experiment_3_1[0], experiment_3_1[1], experiment_3_1[2], experiment_3_1[3])
            experiment_3_2 = Simulator().simulation_step_3(p_frac_exp2)
            plot_regret("STEP 3 - EXP 2", experiment_3_2[0], experiment_3_2[1], experiment_3_2[2])
            plot_reward("STEP 3 - EXP 2", experiment_3_2[0], experiment_3_2[1], experiment_3_2[2], experiment_3_2[3])
            break

        elif choice == 4:
            experiment_4_1 = Simulator().simulation_step_4(p_frac_exp1)
            plot_regret("STEP 4 - EXP 1", experiment_4_1[0], experiment_4_1[1], experiment_4_1[2])
            plot_reward("STEP 4 - EXP 1", experiment_4_1[0], experiment_4_1[1], experiment_4_1[2], experiment_4_1[3])
            experiment_4_2 = Simulator().simulation_step_4(p_frac_exp2)
            plot_regret("STEP 4 - EXP 2", experiment_4_2[0], experiment_4_2[1], experiment_4_2[2])
            plot_reward("STEP 4 - EXP 2", experiment_4_2[0], experiment_4_2[1], experiment_4_2[2], experiment_4_1[3])
            break

        elif choice == 5:
            experiment_5_1 = Simulator().simulation_step_5(p_frac_exp1)
            plot_regret_matching("STEP 5 - EXP 1", experiment_5_1[0])
            plot_reward_matching("STEP 5 - EXP 1", experiment_5_1[1], experiment_5_1[2])
            experiment_5_2 = Simulator().simulation_step_5(p_frac_exp2)
            plot_regret_matching("STEP 5 - EXP 2", experiment_5_2[0])
            plot_reward_matching("STEP 5 - EXP 2", experiment_5_2[1], experiment_5_2[2])
            break

        elif choice == 6:
            experiment_6_1 = Simulator().simulation_step_6(p_frac_exp1)
            plot_regret("STEP 6 - EXP 1", experiment_6_1[0], experiment_6_1[1], experiment_6_1[2])
            plot_reward("STEP 6 - EXP 1", experiment_6_1[1], experiment_6_1[2])
            experiment_6_2 = Simulator().simulation_step_6(p_frac_exp2)
            plot_regret("STEP 6 - EXP 1", experiment_6_2[0], experiment_6_2[1], experiment_6_2[2])
            plot_reward("STEP 6 - EXP 1", experiment_6_2[1], experiment_6_2[2])
            break

        elif choice == 7:
            experiment_7_1 = Simulator().simulation_step_7(p_frac_exp1)
            break

        elif choice == 8:
            experiment_8_1 = Simulator().simulation_step_8(p_frac_exp1)
            experiment_8_2 = Simulator().simulation_step_8(p_frac_exp2)
            break

        else:
            print("You entered an invalid step number. Please try again.")


########################################################################################################################


def plot_regret(step, opt, ucb1_rewards_per_exp, ts_rewards_per_exp):
    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(np.mean(opt - ucb1_rewards_per_exp, axis=0)), "b")
    plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_exp, axis=0)), "r")
    plt.legend(["UCB1", "TS"], title=step)
    plt.show()


def plot_reward(step, opt, ucb1_rewards_per_exp, ts_rewards_per_exp, time_horizon):
    plt.figure(1)
    plt.xlabel("t")
    plt.ylabel("Reward")
    plt.plot(np.mean(ucb1_rewards_per_exp, axis=0), "b")
    plt.plot(np.mean(ts_rewards_per_exp, axis=0), "r")
    opt = [opt] * time_horizon
    plt.plot(opt, '--k')
    plt.legend(["UCB1", "TS", "OPTIMAL"], title=step)
    plt.show()


def plot_regret_matching(step, ucb_matching_regret):
    plt.figure(0)
    plt.xlabel('t')
    plt.ylabel('Regret')
    plt.plot(ucb_matching_regret.mean(axis=0), "b")
    plt.title(step)
    plt.show()


def plot_reward_matching(step, opt, ucb_matching_reward):
    plt.figure(1)
    plt.xlabel('t')
    plt.ylabel('Reward')
    plt.plot(np.mean(ucb_matching_reward, axis=0), "b")
    plt.plot(opt, '--k')
    plt.legend(["UCB1", "OPTIMAL"], title=step)
    plt.show()


if __name__ == "__main__":
    main()
