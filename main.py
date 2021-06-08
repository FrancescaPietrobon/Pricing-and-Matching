from Simulator import *
import pandas as pd
from scipy.ndimage.filters import uniform_filter1d


def main():
    promo_fractions_exp1 = np.array([0.7, 0.2, 0.07, 0.03])
    promo_fractions_exp2 = np.array([0.6, 0.25, 0.1, 0.05])

    print("\t\tData Intelligence Applications - A.Y. 2020/2021")
    print("\t\t\t\tPricing and Matching project\n")

    while True:
        choice = int(input("Select the step (1-3-4-5-6-7-8): "))

        if choice == 1:
            revenue, best_price_item1, best_price_item2, matching = Simulator().simulation_step_1(promo_fractions_exp1)
            result = pd.DataFrame(data=matching,
                                  index=["P0", "P1", "P2", "P3"],
                                  columns=["Class1", "Class2", "Class3", "Class4"])
            print("\nExperiment 1: fractions", promo_fractions_exp1,
                  "\nRevenue:", revenue,
                  "\nOptimal price item 1:", best_price_item1,
                  "\nOptimal price item 2:", best_price_item2,
                  "\nOptimal assignment of promos to customer classes:\n", result)

            revenue, best_price_item1, best_price_item2, matching = Simulator().simulation_step_1(promo_fractions_exp2)
            result = pd.DataFrame(data=matching,
                                  index=["P0", "P1", "P2", "P3"],
                                  columns=["Class1", "Class2", "Class3", "Class4"])
            print("\nExperiment 2: fractions", promo_fractions_exp2,
                  "\nRevenue:", revenue,
                  "\nOptimal price item 1:", best_price_item1,
                  "\nOptimal price item 2:", best_price_item2,
                  "\nOptimal assignment of promos to customer classes:\n", result)
            break

        elif choice == 3:
            opt, ucb_rew, ts_rew, time_horizon = Simulator().simulation_step_3(promo_fractions_exp1)
            plot_regret("STEP 3 - EXP 1", opt, ucb_rew, ts_rew)
            plot_reward("STEP 3 - EXP 1", opt, ucb_rew, ts_rew, time_horizon)
            opt, ucb_rew, ts_rew, time_horizon = Simulator().simulation_step_3(promo_fractions_exp2)
            plot_regret("STEP 3 - EXP 2", opt, ucb_rew, ts_rew)
            plot_reward("STEP 3 - EXP 2", opt, ucb_rew, ts_rew, time_horizon)
            break

        elif choice == 4:
            opt, ucb_rew, ts_rew, time_horizon = Simulator().simulation_step_4(promo_fractions_exp1)
            plot_regret("STEP 4 - EXP 1", opt, ucb_rew, ts_rew)
            plot_reward("STEP 4 - EXP 1", opt, ucb_rew, ts_rew, time_horizon)
            opt, ucb_rew, ts_rew, time_horizon = Simulator().simulation_step_4(promo_fractions_exp2)
            plot_regret("STEP 4 - EXP 2",opt, ucb_rew, ts_rew)
            plot_reward("STEP 4 - EXP 2", opt, ucb_rew, ts_rew, time_horizon)
            break

        elif choice == 5:
            # TODO Do not print regret since it is useless in this problem
            opt, ucb_rew, time_horizon = Simulator().simulation_step_5(promo_fractions_exp1)
            #plot_regret_single_learner("LP", "STEP 5 - EXP 1", opt, ucb_rew)
            plot_reward_single_learner("LP", "STEP 5 - EXP 1", opt, ucb_rew, time_horizon)
            opt, ucb_rew, time_horizon = Simulator().simulation_step_5(promo_fractions_exp2)
            #plot_regret_single_learner("LP", "STEP 5 - EXP 2", opt, ucb_rew)
            plot_reward_single_learner("LP", "STEP 5 - EXP 2", opt, ucb_rew, time_horizon)
            break

        elif choice == 6:
            opt, ucb_rew, time_horizon = Simulator().simulation_step_6(promo_fractions_exp1)
            plot_regret_single_learner("UCB1", "STEP 6 - EXP 1", opt, ucb_rew)
            plot_reward_single_learner("UCB1", "STEP 6 - EXP 1", opt, ucb_rew, time_horizon)
            opt, ucb_rew, time_horizon = Simulator().simulation_step_6(promo_fractions_exp2)
            plot_regret_single_learner("UCB1", "STEP 6 - EXP 2", opt, ucb_rew)
            plot_reward_single_learner("UCB1", "STEP 6 - EXP 2", opt, ucb_rew, time_horizon)
            break

        elif choice == 7:
            experiment_7_1 = Simulator().simulation_step_7(promo_fractions_exp1)
            break

        elif choice == 8:
            experiment_8_1 = Simulator().simulation_step_8(promo_fractions_exp1)
            experiment_8_2 = Simulator().simulation_step_8(promo_fractions_exp2)
            break

        else:
            print("You entered an invalid step number. Please try again.")


########################################################################################################################


def plot_regret(step, opt, ucb1_rewards_per_exp, ts_rewards_per_exp):
    plt.figure()
    plt.xlabel("t")
    plt.ylabel("Regret")
    x = np.arange(len(ucb1_rewards_per_exp[0]), dtype=float)

    y_ucb = np.cumsum(np.mean(opt - ucb1_rewards_per_exp, axis=0))
    y_ucb_std = np.std(opt - ucb1_rewards_per_exp, axis=0)
    plt.plot(y_ucb, "b")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([uniform_filter1d(y_ucb - 60 * y_ucb_std, size=30),
                             uniform_filter1d((y_ucb + 60 * y_ucb_std)[::-1], size=30)]),
             alpha=.3, fc='b')

    y_ts = np.cumsum(np.mean(opt - ts_rewards_per_exp, axis=0))
    y_ts_std = np.std(ts_rewards_per_exp, axis=0)
    plt.plot(y_ts, "r")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([uniform_filter1d(y_ts - 60 * y_ts_std, size=30),
                             uniform_filter1d((y_ts + 60 * y_ts_std)[::-1], size=30)]),
             alpha=.3, fc='r')

    plt.ylim(bottom=0.)
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


def plot_regret_single_learner(learner, step, opt, ucb1_rewards_per_exp):
    plt.figure()
    plt.xlabel("t")
    plt.ylabel("Regret")
    x = np.arange(len(ucb1_rewards_per_exp[0]), dtype=float)

    y_ucb = np.cumsum(np.mean(opt - ucb1_rewards_per_exp, axis=0))
    y_ucb_std = np.std(opt - ucb1_rewards_per_exp, axis=0)
    plt.plot(y_ucb, "b")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([uniform_filter1d(y_ucb - 60 * y_ucb_std, size=30),
                             uniform_filter1d((y_ucb + 60 * y_ucb_std)[::-1], size=30)]),
             alpha=.3, fc='b')

    plt.ylim(bottom=0.)
    plt.legend([learner], title=step)
    plt.show()


def plot_reward_single_learner(learner, step, opt, ucb1_rewards_per_exp, time_horizon):
    plt.figure()
    plt.xlabel("t")
    plt.ylabel("Reward")
    plt.plot(np.mean(ucb1_rewards_per_exp, axis=0), "b")
    opt = [opt] * time_horizon
    plt.plot(opt, '--k')
    plt.legend([learner, "OPTIMAL"], title=step)
    plt.show()


if __name__ == "__main__":
    main()
