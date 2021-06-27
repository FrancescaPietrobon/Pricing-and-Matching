from Simulator import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d


def main():
    promo_fractions_exp1 = np.array([0.70, 0.20, 0.07, 0.03])
    promo_fractions_exp2 = np.array([0.25, 0.25, 0.25, 0.25])

    print("\t\tData Intelligence Applications - A.Y. 2020/2021")
    print("\t\t\t\tPricing and Matching project\n")

    while True:
        step = int(input("Select the step (1-3-4-5-6-7-8): "))

        if step == 1:
            exp = int(input("Select the experiment (1-2): "))
            if exp == 1:
                revenue, best_price_item1, best_price_item2, matching = Simulator().simulation_step_1(promo_fractions_exp1)
                result = pd.DataFrame(data=matching,
                                      index=["P0", "P1", "P2", "P3"],
                                      columns=["Class1", "Class2", "Class3", "Class4"])
                print("\nExperiment 1: fractions", promo_fractions_exp1,
                      "\nRevenue:", revenue,
                      "\nOptimal price item 1:", best_price_item1,
                      "\nOptimal price item 2:", best_price_item2,
                      "\nOptimal assignment of promos to customer classes:\n", result)
            elif exp == 2:
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

        elif step == 3:
            exp = int(input("Select the experiment (1-2): "))
            if exp == 1:
                opt, ucb_rew, ts_rew, time_horizon = Simulator().simulation_step_3(promo_fractions_exp1)
                plot_regret("STEP 3 - EXP 1", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
                plot_reward("STEP 3 - EXP 1", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
            elif exp == 2:
                opt, ucb_rew, ts_rew, time_horizon = Simulator().simulation_step_3(promo_fractions_exp2)
                plot_regret("STEP 3 - EXP 2", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
                plot_reward("STEP 3 - EXP 2", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
            break

        elif step == 4:
            exp = int(input("Select the experiment (1-2): "))
            if exp == 1:
                opt, ucb_rew, ts_rew, time_horizon = Simulator().simulation_step_4(promo_fractions_exp1)
                plot_regret("STEP 4 - EXP 1", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
                plot_reward("STEP 4 - EXP 1", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
            elif exp == 2:
                opt, ucb_rew, ts_rew, time_horizon = Simulator().simulation_step_4(promo_fractions_exp2)
                plot_regret("STEP 4 - EXP 2", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
                plot_reward("STEP 4 - EXP 2", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
            break

        elif step == 5:
            exp = int(input("Select the experiment (1-2): "))
            if exp == 1:
                opt, lp_rew, time_horizon = Simulator().simulation_step_5(promo_fractions_exp1)
                plot_reward("STEP 5 - EXP 1", ["Linear Program"], opt, [lp_rew], time_horizon)
            elif exp == 2:
                opt, lp_rew, time_horizon = Simulator().simulation_step_5(promo_fractions_exp2)
                plot_reward("STEP 5 - EXP 2", ["Linear Program"], opt, [lp_rew], time_horizon)
            break

        elif step == 6:
            exp = int(input("Select the experiment (1-2): "))
            if exp == 1:
                opt, ucb_rew, ts_rew, time_horizon = Simulator().simulation_step_6(promo_fractions_exp1)
                plot_regret("STEP 6 - EXP 1", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
                plot_reward("STEP 6 - EXP 1", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
            elif exp == 2:
                opt, ucb_rew, ts_rew, time_horizon = Simulator().simulation_step_6(promo_fractions_exp2)
                plot_regret("STEP 6 - EXP 2", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
                plot_reward("STEP 6 - EXP 2", ["UCB1", "TS"], opt, [ucb_rew, ts_rew], time_horizon)
            break

        elif step == 7:
            exp = int(input("Select the experiment (1-2): "))
            if exp == 1:
                opt, swts_rew, time_horizon = Simulator().simulation_step_7(promo_fractions_exp1)
                plot_regret("STEP 7 - EXP 1", ["Sliding-Window TS"], opt, [swts_rew], time_horizon)
                plot_reward("STEP 7 - EXP 1", ["Sliding-Window TS"], opt, [swts_rew], time_horizon)
            elif exp == 2:
                opt, swts_rew, time_horizon = Simulator().simulation_step_7(promo_fractions_exp2)
                plot_regret("STEP 7 - EXP 2", ["Sliding-Window TS"], opt, [swts_rew], time_horizon)
                plot_reward("STEP 7 - EXP 2", ["Sliding-Window TS"], opt, [swts_rew], time_horizon)
            break

        elif step == 8:
            exp = int(input("Select the experiment (1-2): "))
            if exp == 1:
                opt, ucb_rew, time_horizon = Simulator().simulation_step_8(promo_fractions_exp1)
                plot_regret("STEP 8 - EXP 1", ["Change-Detection UCB1"], opt, [ucb_rew], time_horizon)
                plot_reward("STEP 8 - EXP 1", ["Change-Detection UCB1"], opt, [ucb_rew], time_horizon)
            elif exp == 2:
                opt, ucb_rew, time_horizon = Simulator().simulation_step_8(promo_fractions_exp1)
                plot_regret("STEP 8 - EXP 2", ["Change-Detection UCB1"], opt, [ucb_rew], time_horizon)
                plot_reward("STEP 8 - EXP 2", ["Change-Detection UCB1"], opt, [ucb_rew], time_horizon)
            break

        else:
            print("You entered an invalid step number. Please try again.")


########################################################################################################################


def plot_regret(step, learners, opt, rewards_per_exp, time_horizon):
    plt.figure()
    plt.xlabel("t")
    plt.ylabel("Regret")

    x = np.arange(len(rewards_per_exp[0][0]), dtype=float)
    colours = ['b', 'r']

    for learner in range(len(learners)):
        y = np.cumsum(np.mean(opt - rewards_per_exp[learner], axis=0))
        y_std = np.std(opt - rewards_per_exp[learner], axis=0)
        plt.plot(y, colours[learner])
        if time_horizon == 365:
            plt.fill(np.concatenate([x, x[::-1]]),
                     np.concatenate([uniform_filter1d(y - 30 * y_std, size=30),
                                     uniform_filter1d((y + 30 * y_std)[::-1], size=30)]),
                     alpha=.3, fc=colours[learner])
        elif time_horizon == 5000:
            plt.fill(np.concatenate([x, x[::-1]]),
                     np.concatenate([uniform_filter1d(y - 150 * y_std, size=60),
                                     uniform_filter1d((y + 150 * y_std)[::-1], size=60)]),
                     alpha=.3, fc=colours[learner])

    plt.ylim(bottom=0.)
    plt.legend(learners, title=step)
    plt.show()


def plot_reward(step, learners, opt, rewards_per_exp, time_horizon):
    plt.figure()
    plt.xlabel("t")
    plt.ylabel("Reward")
    colours = ['b', 'r']

    for learner in range(len(learners)):
        plt.plot(np.mean(rewards_per_exp[learner], axis=0), colours[learner])

    ns_steps = ["STEP 7", "STEP 8"]
    if not any(x in step for x in ns_steps):
        opt = [opt] * time_horizon
    plt.plot(opt, '--k')
    learners.append("OPTIMAL")
    plt.legend(learners, title=step)
    plt.show()


if __name__ == "__main__":
    main()
