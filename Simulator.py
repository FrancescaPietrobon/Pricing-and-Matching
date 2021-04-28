from Group import *
from Data import *
from Item import *
from LP_optimization import *
from UCB1 import *
from UCB_matching import *
from TS_Learner import *
from Environment import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import normal, binomial
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestRegressor

np.random.seed(1234)


class Simulator:
    def __init__(self):
        # Pay attention that in other simulations we need to overwrite group data
        self.customers_groups = np.array([Group(1), Group(2), Group(3), Group(4)])
        self.item1 = Item("Apple Watch", 300, 88)
        self.item2 = Item("Personalized wristband", 50, 16)
        self.discount_p1 = 0.1
        self.discount_p2 = 0.2
        self.discount_p3 = 0.5

    def simulation_step_1(self, p0_frac, p1_frac, p2_frac, p3_frac):
        # Creating the Data object to get the actual numbers from the Google Module
        data = Data()

        # Daily number of customers per class = Gaussian TODO: are sigmas correct?
        daily_customers = np.array([int(normal(data.get_n(1), 12)),
                                    int(normal(data.get_n(2), 14)),
                                    int(normal(data.get_n(3), 16)),
                                    int(normal(data.get_n(4), 17))])

        # Number of promo codes available daily (fixed fraction of the daily number of customers)
        daily_promos = [int(sum(daily_customers) * p0_frac),
                        int(sum(daily_customers) * p1_frac),
                        int(sum(daily_customers) * p2_frac),
                        int(sum(daily_customers) * p3_frac)]

        # Probability that a customer of a class buys the second item given the first + each promo
        # rows: promo code (0: P0, 1: P1, 2: P2, 3: P3)
        # columns: customer group (0: group1, 1: group2, 2: group3, 3: group4)
        prob_buy_item21 = np.array([  # Promo code P0
                                    [binomial(daily_customers[0], data.get_i21_p0_param(1)) / daily_customers[0],
                                     binomial(daily_customers[1], data.get_i21_p0_param(2)) / daily_customers[1],
                                     binomial(daily_customers[2], data.get_i21_p0_param(3)) / daily_customers[2],
                                     binomial(daily_customers[3], data.get_i21_p0_param(4)) / daily_customers[3]],
                                    # Promo code P1
                                    [binomial(daily_customers[0], data.get_i21_p1_param(1)) / daily_customers[0],
                                     binomial(daily_customers[1], data.get_i21_p1_param(2)) / daily_customers[1],
                                     binomial(daily_customers[2], data.get_i21_p1_param(3)) / daily_customers[2],
                                     binomial(daily_customers[3], data.get_i21_p1_param(4)) / daily_customers[3]],
                                    # Promo code P2
                                    [binomial(daily_customers[0], data.get_i21_p2_param(1)) / daily_customers[0],
                                     binomial(daily_customers[1], data.get_i21_p2_param(2)) / daily_customers[1],
                                     binomial(daily_customers[2], data.get_i21_p2_param(3)) / daily_customers[2],
                                     binomial(daily_customers[3], data.get_i21_p2_param(4)) / daily_customers[3]],
                                    # Promo code P3
                                    [binomial(daily_customers[0], data.get_i21_p3_param(1)) / daily_customers[0],
                                     binomial(daily_customers[1], data.get_i21_p3_param(2)) / daily_customers[1],
                                     binomial(daily_customers[2], data.get_i21_p3_param(3)) / daily_customers[2],
                                     binomial(daily_customers[3], data.get_i21_p3_param(4)) / daily_customers[3]]
                                ])

        # Linear optimization algorithm to find the best matching
        return LP(self.item2.get_price(), self.discount_p1, self.discount_p2, self.discount_p3,
                  prob_buy_item21[0][0], prob_buy_item21[0][1], prob_buy_item21[0][2], prob_buy_item21[0][3],
                  prob_buy_item21[1][0], prob_buy_item21[1][1], prob_buy_item21[1][2], prob_buy_item21[1][3],
                  prob_buy_item21[2][0], prob_buy_item21[2][1], prob_buy_item21[2][2], prob_buy_item21[2][3],
                  prob_buy_item21[3][0], prob_buy_item21[3][1], prob_buy_item21[3][2], prob_buy_item21[3][3],
                  daily_promos[0], daily_promos[1], daily_promos[2], daily_promos[3],
                  daily_customers[0], daily_customers[1], daily_customers[2], daily_customers[3])

########################################################################################################################

    def simulation_step_3(self):
        # Time horizon
        T = 1000

        # Number of arms (computed as np.ceil((T * np.log(T))**(1/4)).astype(int))
        n_arms = 9

        # Candidate prices (one per arm) - The central one (€300) is taken by step 1
        prices = np.array([50, 100, 150, 200, 300, 400, 450, 500, 550])

        # Conversion rates for item 1 (one per arm) - The central one (0.43) is taken by step 1
        conversion_rates_item1 = np.array([0.9, 0.75, 0.57, 0.44, 0.43, 0.42, 0.13, 0.10, 0.05])

        # Number of daily customers (one per class) - Taken by step 1
        daily_customers = np.array([380, 220, 267, 124])

        # Promo assigment (row: promo; column: customer class) - Taken by step 1
        weights = np.array([[0.92553191, 1, 0, 1],
                            [0, 0, 0.74339623, 0],
                            [0, 0, 0.25660377, 0],
                            [0.07446809, 0, 0, 0]])

        # Conversion rates for item 2 given item 1 and promo (one row per class per promo; one element per arm)
        conversion_rates_item21 = np.array([    # Promo P0, Classes 1-2-3-4
                                                [[0.52, 0.47, 0.43, 0.38, 0.36, 0.34, 0.31, 0.30, 0.27],
                                                 [0.39, 0.34, 0.29, 0.22, 0.20, 0.19, 0.15, 0.13, 0.10],
                                                 [0.46, 0.40, 0.36, 0.32, 0.29, 0.27, 0.20, 0.17, 0.13],
                                                 [0.33, 0.29, 0.25, 0.21, 0.15, 0.13, 0.11, 0.10, 0.07]],
                                                # Promo P1, Classes 1-2-3-4
                                                [[0.64, 0.59, 0.53, 0.48, 0.41, 0.39, 0.35, 0.31, 0.26],
                                                 [0.45, 0.39, 0.37, 0.29, 0.26, 0.23, 0.19, 0.17, 0.12],
                                                 [0.32, 0.29, 0.27, 0.26, 0.25, 0.20, 0.12, 0.08, 0.05],
                                                 [0.37, 0.30, 0.25, 0.24, 0.23, 0.19, 0.16, 0.15, 0.10]],
                                                # Promo P2, Classes 1-2-3-4
                                                [[0.74, 0.62, 0.57, 0.50, 0.44, 0.42, 0.37, 0.30, 0.24],
                                                 [0.42, 0.39, 0.35, 0.31, 0.27, 0.23, 0.21, 0.20, 0.16],
                                                 [0.54, 0.49, 0.45, 0.35, 0.32, 0.29, 0.22, 0.21, 0.17],
                                                 [0.36, 0.30, 0.27, 0.21, 0.19, 0.17, 0.13, 0.10, 0.06]],
                                                # Promo P3, Classes 1-2-3-4
                                                [[0.95, 0.92, 0.85, 0.79, 0.76, 0.69, 0.58, 0.50, 0.43],
                                                 [0.83, 0.79, 0.73, 0.68, 0.63, 0.56, 0.53, 0.47, 0.40],
                                                 [0.88, 0.80, 0.71, 0.64, 0.61, 0.58, 0.51, 0.43, 0.38],
                                                 [0.61, 0.54, 0.49, 0.47, 0.46, 0.44, 0.35, 0.31, 0.27]]
                                            ])

        # Computing the objective array (one element per arm)
        objective = np.zeros(n_arms)
        for i in range(n_arms):
            objective[i] = prices[i] * conversion_rates_item1[i] * sum(daily_customers) +\
                           self.item2.get_price() * conversion_rates_item1[i] * (
                           daily_customers[0] * conversion_rates_item21[0][0][i] * weights[0][0] +
                           daily_customers[1] * conversion_rates_item21[0][1][i] * weights[0][1] +
                           daily_customers[2] * conversion_rates_item21[0][2][i] * weights[0][2] +
                           daily_customers[3] * conversion_rates_item21[0][3][i] * weights[0][3] +
                           daily_customers[0] * conversion_rates_item21[1][0][i] * weights[1][0] * (1-self.discount_p1) +
                           daily_customers[1] * conversion_rates_item21[1][1][i] * weights[1][1] * (1-self.discount_p1) +
                           daily_customers[2] * conversion_rates_item21[1][2][i] * weights[1][2] * (1-self.discount_p1) +
                           daily_customers[3] * conversion_rates_item21[1][3][i] * weights[1][3] * (1-self.discount_p1) +
                           daily_customers[0] * conversion_rates_item21[2][0][i] * weights[2][0] * (1-self.discount_p2) +
                           daily_customers[1] * conversion_rates_item21[2][1][i] * weights[2][1] * (1-self.discount_p2) +
                           daily_customers[2] * conversion_rates_item21[2][2][i] * weights[2][2] * (1-self.discount_p2) +
                           daily_customers[3] * conversion_rates_item21[2][3][i] * weights[2][3] * (1-self.discount_p2) +
                           daily_customers[0] * conversion_rates_item21[3][0][i] * weights[3][0] * (1-self.discount_p3) +
                           daily_customers[1] * conversion_rates_item21[3][1][i] * weights[3][1] * (1-self.discount_p3) +
                           daily_customers[2] * conversion_rates_item21[3][2][i] * weights[3][2] * (1-self.discount_p3) +
                           daily_customers[3] * conversion_rates_item21[3][3][i] * weights[3][3] * (1-self.discount_p3))

        # Storing the optimal objective value to compute the regret later
        opt_env2 = max(objective)
        normalized_objective = objective / np.linalg.norm(objective)
        opt_env1 = max(normalized_objective)

        # Launching the experiments, using both UCB1 and Thompson Sampling
        # Two different approaches for the environment are used (see Environment class)
        n_experiments = 100
        ucb1_rewards_per_experiment_env1 = []
        ts_rewards_per_experiment_env1 = []
        ucb1_rewards_per_experiment_env2 = []
        ts_rewards_per_experiment_env2 = []

        for e in range(n_experiments):
            env1 = Environment_First(n_arms=n_arms, probabilities=normalized_objective)
            env2 = Environment_Second(n_arms=n_arms, probabilities=conversion_rates_item1, candidates=prices)
            ucb1_learner_env1 = UCB1(n_arms=n_arms)
            ucb1_learner_env2 = UCB1(n_arms=n_arms)
            ts_learner_env1 = TS_Learner(n_arms=n_arms)
            ts_learner_env2 = TS_Learner(n_arms=n_arms)

            for t in range(0, T):
                # UCB1 Learner
                pulled_arm = ucb1_learner_env1.pull_arm()
                reward = env1.round(pulled_arm)
                ucb1_learner_env1.update(pulled_arm, reward)

                pulled_arm = ucb1_learner_env2.pull_arm()
                reward = env2.round(pulled_arm)
                ucb1_learner_env2.update(pulled_arm, reward)

                # Thompson Sampling Learner
                pulled_arm = ts_learner_env1.pull_arm()
                reward = env1.round(pulled_arm)
                ts_learner_env1.update(pulled_arm, reward)

                pulled_arm = ts_learner_env2.pull_arm()
                reward = env2.round(pulled_arm)
                ts_learner_env2.update(pulled_arm, reward)

            ucb1_rewards_per_experiment_env1.append(ucb1_learner_env1.collected_rewards)
            ts_rewards_per_experiment_env1.append(ts_learner_env1.collected_rewards)
            ucb1_rewards_per_experiment_env2.append(ucb1_learner_env2.collected_rewards)
            ts_rewards_per_experiment_env2.append(ts_learner_env2.collected_rewards)

        # Rescaling the rewards (only in the case of the second environment)
        ucb1_rewards_per_experiment_env2 = [x * max(prices) for x in ucb1_rewards_per_experiment_env2]
        ts_rewards_per_experiment_env2 = [x * max(prices) for x in ts_rewards_per_experiment_env2]

        # Plotting the regret and the reward related to the first environment
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(opt_env1 - ts_rewards_per_experiment_env1, axis=0)), "r")
        plt.plot(np.cumsum(np.mean(opt_env1 - ucb1_rewards_per_experiment_env1, axis=0)), "b")
        plt.legend(["TS", "UCB1"], title="STEP 3 - ENV1")
        plt.show()

        plt.figure(1)
        plt.xlabel("t")
        plt.ylabel("Reward")
        plt.plot(np.mean(ts_rewards_per_experiment_env1, axis=0), "r")
        plt.plot(np.mean(ucb1_rewards_per_experiment_env1, axis=0), "b")
        plt.legend(["TS-ENV1", "UCB1-ENV1"], title="STEP 3 - ENV1")
        plt.show()

        # Plotting the regret and the reward related to the second environment
        plt.figure(2)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(opt_env2 - ts_rewards_per_experiment_env2, axis=0)), "g")
        plt.plot(np.cumsum(np.mean(opt_env2 - ucb1_rewards_per_experiment_env2, axis=0)), "y")
        plt.legend(["TS", "UCB1"], title="STEP 3 - ENV2")
        plt.show()

        plt.figure(3)
        plt.xlabel("t")
        plt.ylabel("Reward")
        plt.plot(np.mean(ts_rewards_per_experiment_env2, axis=0), "g")
        plt.plot(np.mean(ucb1_rewards_per_experiment_env2, axis=0), "y")
        plt.legend(["TS", "UCB1"], title="STEP 3 - ENV2")
        plt.show()

########################################################################################################################

    def simulation_step_4(self):
        # Time horizon
        T = 1000

        # Number of arms (computed as np.ceil((T * np.log(T))**(1/4)).astype(int))
        n_arms = 9

        # Candidate prices (one per arm) - The central one (€300) is taken by step 1
        prices = np.array([50, 100, 150, 200, 300, 400, 450, 500, 550])

        # Conversion rates for item 1 (one per arm) - The central one (0.43) is taken by step 1
        conversion_rates_item1 = np.array([0.9, 0.75, 0.57, 0.44, 0.43, 0.42, 0.13, 0.10, 0.05])

        # Promo assigment (row: promo; column: customer class) - Taken by step 1
        weights = np.array([[0.92553191, 1, 0, 1],
                            [0, 0, 0.74339623, 0],
                            [0, 0, 0.25660377, 0],
                            [0.07446809, 0, 0, 0]])

        # Conversion rates for item 2 given item 1 and promo (one row per class per promo; one element per arm)
        # We generate each one of them from a Gaussian distribution in which the mean is the corresponding
        # conversion rate fixed in Step 3, and the variance is fixed.
        conversion_rates_item21_means = np.array([  # Promo P0, Classes 1-2-3-4
                                                    [[0.52, 0.47, 0.43, 0.38, 0.36, 0.34, 0.31, 0.30, 0.27],
                                                     [0.39, 0.34, 0.29, 0.22, 0.20, 0.19, 0.15, 0.13, 0.10],
                                                     [0.46, 0.40, 0.36, 0.32, 0.29, 0.27, 0.20, 0.17, 0.13],
                                                     [0.33, 0.29, 0.25, 0.21, 0.15, 0.13, 0.11, 0.10, 0.07]],
                                                    # Promo P1, Classes 1-2-3-4
                                                    [[0.64, 0.59, 0.53, 0.48, 0.41, 0.39, 0.35, 0.31, 0.26],
                                                     [0.45, 0.39, 0.37, 0.29, 0.26, 0.23, 0.19, 0.17, 0.12],
                                                     [0.32, 0.29, 0.27, 0.26, 0.25, 0.20, 0.12, 0.08, 0.05],
                                                     [0.37, 0.30, 0.25, 0.24, 0.23, 0.19, 0.16, 0.15, 0.10]],
                                                    # Promo P2, Classes 1-2-3-4
                                                    [[0.74, 0.62, 0.57, 0.50, 0.44, 0.42, 0.37, 0.30, 0.24],
                                                     [0.42, 0.39, 0.35, 0.31, 0.27, 0.23, 0.21, 0.20, 0.16],
                                                     [0.54, 0.49, 0.45, 0.35, 0.32, 0.29, 0.22, 0.21, 0.17],
                                                     [0.36, 0.30, 0.27, 0.21, 0.19, 0.17, 0.13, 0.10, 0.06]],
                                                    # Promo P3, Classes 1-2-3-4
                                                    [[0.95, 0.92, 0.85, 0.79, 0.76, 0.69, 0.58, 0.50, 0.43],
                                                     [0.83, 0.79, 0.73, 0.68, 0.63, 0.56, 0.53, 0.47, 0.40],
                                                     [0.88, 0.80, 0.71, 0.64, 0.61, 0.58, 0.51, 0.43, 0.38],
                                                     [0.61, 0.54, 0.49, 0.47, 0.46, 0.44, 0.35, 0.31, 0.27]]
                                                ])

        conversion_rates_item21 = normal(conversion_rates_item21_means, 0.025, (4, 4, 9))

        # Computing the objective array (one element per arm) TODO we are not using daily_customers
        objective = np.zeros(n_arms)
        for i in range(n_arms):
            objective[i] = prices[i] * conversion_rates_item1[i] + \
                           self.item2.get_price() * conversion_rates_item1[i] * (
                           conversion_rates_item21[0][0][i] * weights[0][0] +
                           conversion_rates_item21[0][1][i] * weights[0][1] +
                           conversion_rates_item21[0][2][i] * weights[0][2] +
                           conversion_rates_item21[0][3][i] * weights[0][3] +
                           conversion_rates_item21[1][0][i] * weights[1][0] * (1 - self.discount_p1) +
                           conversion_rates_item21[1][1][i] * weights[1][1] * (1 - self.discount_p1) +
                           conversion_rates_item21[1][2][i] * weights[1][2] * (1 - self.discount_p1) +
                           conversion_rates_item21[1][3][i] * weights[1][3] * (1 - self.discount_p1) +
                           conversion_rates_item21[2][0][i] * weights[2][0] * (1 - self.discount_p2) +
                           conversion_rates_item21[2][1][i] * weights[2][1] * (1 - self.discount_p2) +
                           conversion_rates_item21[2][2][i] * weights[2][2] * (1 - self.discount_p2) +
                           conversion_rates_item21[2][3][i] * weights[2][3] * (1 - self.discount_p2) +
                           conversion_rates_item21[3][0][i] * weights[3][0] * (1 - self.discount_p3) +
                           conversion_rates_item21[3][1][i] * weights[3][1] * (1 - self.discount_p3) +
                           conversion_rates_item21[3][2][i] * weights[3][2] * (1 - self.discount_p3) +
                           conversion_rates_item21[3][3][i] * weights[3][3] * (1 - self.discount_p3))

        # Storing the optimal objective value to compute the regret later
        opt_env2 = max(objective)
        normalized_objective = objective / np.linalg.norm(objective)
        opt_env1 = max(normalized_objective)

        # Launching the experiments, using both UCB1 and Thompson Sampling
        # Two different approaches for the environment are used (see Environment class)
        n_experiments = 100
        ucb1_rewards_per_experiment_env1 = []
        ts_rewards_per_experiment_env1 = []
        ucb1_rewards_per_experiment_env2 = []
        ts_rewards_per_experiment_env2 = []

        for e in range(n_experiments):
            env1 = Environment_First(n_arms=n_arms, probabilities=normalized_objective)
            env2 = Environment_Second(n_arms=n_arms, probabilities=conversion_rates_item1, candidates=prices)
            ucb1_learner_env1 = UCB1(n_arms=n_arms)
            ucb1_learner_env2 = UCB1(n_arms=n_arms)
            ts_learner_env1 = TS_Learner(n_arms=n_arms)
            ts_learner_env2 = TS_Learner(n_arms=n_arms)

            for t in range(0, T):
                # UCB1 Learner
                pulled_arm = ucb1_learner_env1.pull_arm()
                reward = env1.round(pulled_arm)
                ucb1_learner_env1.update(pulled_arm, reward)

                pulled_arm = ucb1_learner_env2.pull_arm()
                reward = env2.round(pulled_arm)
                ucb1_learner_env2.update(pulled_arm, reward)

                # Thompson Sampling Learner
                pulled_arm = ts_learner_env1.pull_arm()
                reward = env1.round(pulled_arm)
                ts_learner_env1.update(pulled_arm, reward)

                pulled_arm = ts_learner_env2.pull_arm()
                reward = env2.round(pulled_arm)
                ts_learner_env2.update(pulled_arm, reward)

            ucb1_rewards_per_experiment_env1.append(ucb1_learner_env1.collected_rewards)
            ts_rewards_per_experiment_env1.append(ts_learner_env1.collected_rewards)
            ucb1_rewards_per_experiment_env2.append(ucb1_learner_env2.collected_rewards)
            ts_rewards_per_experiment_env2.append(ts_learner_env2.collected_rewards)

        # Rescaling the rewards (only in the case of the second environment)
        ucb1_rewards_per_experiment_env2 = [x * max(prices) for x in ucb1_rewards_per_experiment_env2]
        ts_rewards_per_experiment_env2 = [x * max(prices) for x in ts_rewards_per_experiment_env2]

        # Plotting the regret and the reward related to the first environment
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(opt_env1 - ts_rewards_per_experiment_env1, axis=0)), "r")
        plt.plot(np.cumsum(np.mean(opt_env1 - ucb1_rewards_per_experiment_env1, axis=0)), "b")
        plt.legend(["TS", "UCB1"], title="STEP 4 - ENV1")
        plt.show()

        plt.figure(1)
        plt.xlabel("t")
        plt.ylabel("Reward")
        plt.plot(np.mean(ts_rewards_per_experiment_env1, axis=0), "r")
        plt.plot(np.mean(ucb1_rewards_per_experiment_env1, axis=0), "b")
        plt.legend(["TS-ENV1", "UCB1-ENV1"], title="STEP 4 - ENV1")
        plt.show()

        # Plotting the regret and the reward related to the second environment
        plt.figure(2)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(opt_env2 - ts_rewards_per_experiment_env2, axis=0)), "g")
        plt.plot(np.cumsum(np.mean(opt_env2 - ucb1_rewards_per_experiment_env2, axis=0)), "y")
        plt.legend(["TS", "UCB1"], title="STEP 4 - ENV2")
        plt.show()

        plt.figure(3)
        plt.xlabel("t")
        plt.ylabel("Reward")
        plt.plot(np.mean(ts_rewards_per_experiment_env2, axis=0), "g")
        plt.plot(np.mean(ucb1_rewards_per_experiment_env2, axis=0), "y")
        plt.legend(["TS", "UCB1"], title="STEP 4 - ENV2")
        plt.show()

########################################################################################################################

    def simulation_step_5(self):

        #Dataset
        #      c1     c2     c3      c4      P0     P1     P2     P3             T (c.)
        #      1      0       0       0      0      1       0     0             c...

        # Creating the Data object to get the actual numbers from the Google Module
        data = Data()
        dataset = pd.DataFrame(columns=['c1', 'c2', 'c3', 'c4', 'P0', 'P1', 'P2', 'P3', 'TARGET'])

        sum_daily_customer = np.zeros(4)

        for i in range(365):

            # Daily number of customers per class = Gaussian TODO: are sigmas correct?
            daily_customers = np.array([int(normal(data.get_n(1), 12)),
                                        int(normal(data.get_n(2), 14)),
                                        int(normal(data.get_n(3), 16)),
                                        int(normal(data.get_n(4), 17))])

            # Probability that a customer of a class buys the second item given the first + each promo
            # rows: promo code (0: P0, 1: P1, 2: P2, 3: P3)
            # columns: customer group (0: group1, 1: group2, 2: group3, 3: group4)
            prob_buy_item21 = np.array([  # Promo code P0
                [binomial(daily_customers[0], data.get_i21_p0_param(1)) / daily_customers[0],
                 binomial(daily_customers[1], data.get_i21_p0_param(2)) / daily_customers[1],
                 binomial(daily_customers[2], data.get_i21_p0_param(3)) / daily_customers[2],
                 binomial(daily_customers[3], data.get_i21_p0_param(4)) / daily_customers[3]],
                # Promo code P1
                [binomial(daily_customers[0], data.get_i21_p1_param(1)) / daily_customers[0],
                 binomial(daily_customers[1], data.get_i21_p1_param(2)) / daily_customers[1],
                 binomial(daily_customers[2], data.get_i21_p1_param(3)) / daily_customers[2],
                 binomial(daily_customers[3], data.get_i21_p1_param(4)) / daily_customers[3]],
                # Promo code P2
                [binomial(daily_customers[0], data.get_i21_p2_param(1)) / daily_customers[0],
                 binomial(daily_customers[1], data.get_i21_p2_param(2)) / daily_customers[1],
                 binomial(daily_customers[2], data.get_i21_p2_param(3)) / daily_customers[2],
                 binomial(daily_customers[3], data.get_i21_p2_param(4)) / daily_customers[3]],
                # Promo code P3
                [binomial(daily_customers[0], data.get_i21_p3_param(1)) / daily_customers[0],
                 binomial(daily_customers[1], data.get_i21_p3_param(2)) / daily_customers[1],
                 binomial(daily_customers[2], data.get_i21_p3_param(3)) / daily_customers[2],
                 binomial(daily_customers[3], data.get_i21_p3_param(4)) / daily_customers[3]]
            ])

            dataset.loc[len(dataset)] = [1, 0, 0, 0, 1, 0, 0, 0, prob_buy_item21[0][0]]
            dataset.loc[len(dataset)] = [0, 1, 0, 0, 1, 0, 0, 0, prob_buy_item21[0][1]]
            dataset.loc[len(dataset)] = [0, 0, 1, 0, 1, 0, 0, 0, prob_buy_item21[0][2]]
            dataset.loc[len(dataset)] = [0, 0, 0, 1, 1, 0, 0, 0, prob_buy_item21[0][3]]
            dataset.loc[len(dataset)] = [1, 0, 0, 0, 0, 1, 0, 0, prob_buy_item21[1][0]]
            dataset.loc[len(dataset)] = [0, 1, 0, 0, 0, 1, 0, 0, prob_buy_item21[1][1]]
            dataset.loc[len(dataset)] = [0, 0, 1, 0, 0, 1, 0, 0, prob_buy_item21[1][2]]
            dataset.loc[len(dataset)] = [0, 0, 0, 1, 0, 1, 0, 0, prob_buy_item21[1][3]]
            dataset.loc[len(dataset)] = [1, 0, 0, 0, 0, 0, 1, 0, prob_buy_item21[2][0]]
            dataset.loc[len(dataset)] = [0, 1, 0, 0, 0, 0, 1, 0, prob_buy_item21[2][1]]
            dataset.loc[len(dataset)] = [0, 0, 1, 0, 0, 0, 1, 0, prob_buy_item21[2][2]]
            dataset.loc[len(dataset)] = [0, 0, 0, 1, 0, 0, 1, 0, prob_buy_item21[2][3]]
            dataset.loc[len(dataset)] = [1, 0, 0, 0, 0, 0, 0, 1, prob_buy_item21[3][0]]
            dataset.loc[len(dataset)] = [0, 1, 0, 0, 0, 0, 0, 1, prob_buy_item21[3][1]]
            dataset.loc[len(dataset)] = [0, 0, 1, 0, 0, 0, 0, 1, prob_buy_item21[3][2]]
            dataset.loc[len(dataset)] = [0, 0, 0, 1, 0, 0, 0, 1, prob_buy_item21[3][3]]

            sum_daily_customer += daily_customers

        target = 'TARGET'
        features = dataset.columns[dataset.columns != target]

        X = dataset[features].values
        y = dataset[target].values

        regressor = RandomForestRegressor(n_estimators=100, random_state=1234)
        regressor.fit(X, y)

        prediction_set = pd.DataFrame(columns=['c1', 'c2', 'c3', 'c4', 'P0', 'P1', 'P2', 'P3'])
        prediction_set.loc[len(prediction_set)] = [1, 0, 0, 0, 1, 0, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 1, 0, 0, 1, 0, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 1, 0, 1, 0, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 0, 1, 1, 0, 0, 0]
        prediction_set.loc[len(prediction_set)] = [1, 0, 0, 0, 0, 1, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 1, 0, 0, 0, 1, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 1, 0, 0, 1, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 0, 1, 0, 1, 0, 0]
        prediction_set.loc[len(prediction_set)] = [1, 0, 0, 0, 0, 0, 1, 0]
        prediction_set.loc[len(prediction_set)] = [0, 1, 0, 0, 0, 0, 1, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 1, 0, 0, 0, 1, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 0, 1, 0, 0, 1, 0]
        prediction_set.loc[len(prediction_set)] = [1, 0, 0, 0, 0, 0, 0, 1]
        prediction_set.loc[len(prediction_set)] = [0, 1, 0, 0, 0, 0, 0, 1]
        prediction_set.loc[len(prediction_set)] = [0, 0, 1, 0, 0, 0, 0, 1]
        prediction_set.loc[len(prediction_set)] = [0, 0, 0, 1, 0, 0, 0, 1]
        res = regressor.predict(prediction_set.values)

        print(res)

        # p = np.array([[1/4, 1, 1/4], [1/2, 1/4, 1/4], [1/4, 1/4, 1]])
        # rows: promo codes; columns: customers - values: price_item2 * (1-discount) * conversion_rate_item2_given1_promo (average)
        #           cust1 (c1)   cust2 (c1)   cust3 (c1)   cust4 (c2)   cust5 (c3)  ...
        #   P0
        #   P0
        #   P0
        #   P0
        #   P1
        #   P1
        #   P2
        #   P3
        #   ...

        price_item2 = 50
        prob_buy_item21 = np.array([[res[0], res[1], res[2], res[3]],
                                    [res[4], res[5], res[6], res[7]],
                                    [res[8], res[9], res[10], res[11]],
                                    [res[12], res[13], res[14], res[15]]])
        discount_p1 = 0.1
        discount_p2 = 0.2
        discount_p3 = 0.5
        avg_daily_customer = (sum_daily_customer / 365).astype(int)
        n_promos = sum(avg_daily_customer)
        p = np.zeros((n_promos, sum(avg_daily_customer)))
        # p.fill(-1e6)
        for row_index in range(n_promos):
            for column_index in range(sum(avg_daily_customer)):
                if row_index < 0.7 * n_promos:
                    if column_index < avg_daily_customer[0]:
                        p[row_index, column_index] = price_item2 * prob_buy_item21[0][0]
                    elif column_index < avg_daily_customer[1]:
                        p[row_index, column_index] = price_item2 * prob_buy_item21[0][1]
                    elif column_index < avg_daily_customer[2]:
                        p[row_index, column_index] = price_item2 * prob_buy_item21[0][2]
                    else:
                        p[row_index, column_index] = price_item2 * prob_buy_item21[0][3]
                elif row_index < 0.9 * n_promos:
                    if column_index < avg_daily_customer[0]:
                        p[row_index, column_index] = price_item2 * (1 - discount_p1) * prob_buy_item21[1][0]
                    elif column_index < avg_daily_customer[1]:
                        p[row_index, column_index] = price_item2 * (1 - discount_p1) * prob_buy_item21[1][1]
                    elif column_index < avg_daily_customer[2]:
                        p[row_index, column_index] = price_item2 * (1 - discount_p1) * prob_buy_item21[1][2]
                    else:
                        p[row_index, column_index] = price_item2 * (1 - discount_p1) * prob_buy_item21[1][3]
                elif row_index < 0.97 * n_promos:
                    if column_index < avg_daily_customer[0]:
                        p[row_index, column_index] = price_item2 * (1 - discount_p2) * prob_buy_item21[2][0]
                    elif column_index < avg_daily_customer[1]:
                        p[row_index, column_index] = price_item2 * (1 - discount_p2) * prob_buy_item21[2][1]
                    elif column_index < avg_daily_customer[2]:
                        p[row_index, column_index] = price_item2 * (1 - discount_p2) * prob_buy_item21[2][2]
                    else:
                        p[row_index, column_index] = price_item2 * (1 - discount_p2) * prob_buy_item21[2][3]
                else:
                    if column_index < avg_daily_customer[0]:
                        p[row_index, column_index] = price_item2 * (1 - discount_p3) * prob_buy_item21[3][0]
                    elif column_index < avg_daily_customer[1]:
                        p[row_index, column_index] = price_item2 * (1 - discount_p3) * prob_buy_item21[3][1]
                    elif column_index < avg_daily_customer[2]:
                        p[row_index, column_index] = price_item2 * (1 - discount_p3) * prob_buy_item21[3][2]
                    else:
                        p[row_index, column_index] = price_item2 * (1 - discount_p3) * prob_buy_item21[3][3]

        #p = p / np.linalg.norm(p)
        p = p / p.max()
        opt = linear_sum_assignment(-p)
        n_exp = 10
        T = 1000
        regret_ucb = np.zeros((n_exp, T))
        reward_ucb = []
        for e in range(n_exp):
            learner = UCB_Matching(p.size, *p.shape)
            print(e+1)
            rew_UCB = []
            opt_rew = []
            env = Environment_Second(p.size, p, )
            for t in range(T):
                pulled_arms = learner.pull_arm()
                rewards = env.round(pulled_arms)
                learner.update(pulled_arms, rewards)
                rew_UCB.append(rewards.sum())
                opt_rew.append(p[opt].sum())
            regret_ucb[e, :] = np.cumsum(opt_rew) - np.cumsum(rew_UCB)
            reward_ucb.append(rew_UCB)

        plt.figure(0)
        plt.plot(regret_ucb.mean(axis=0))
        plt.title('STEP 5')
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.show()

        plt.figure(1)
        plt.plot(np.mean(reward_ucb, axis=0))
        plt.title('STEP 5')
        plt.ylabel('Reward')
        plt.xlabel('t')
        plt.show()

########################################################################################################################

    def simulation_step_6(self):
        # Dataset
        #      c1     c2     c3      c4      P0     P1     P2     P3             T (c.)
        #      1      0       0       0      0      1       0     0             c...

        # Creating the Data object to get the actual numbers from the Google Module
        data = Data()
        dataset = pd.DataFrame(columns=['c1', 'c2', 'c3', 'c4', 'P0', 'P1', 'P2', 'P3', 'TARGET'])
        dataset2 = pd.DataFrame(columns=['c1', 'c2', 'c3', 'c4', 'TARGET'])

        sum_daily_customer = np.zeros(4)


        for i in range(365):
            # Daily number of customers per class = Gaussian TODO: are sigmas correct?
            daily_customers = np.array([int(normal(data.get_n(1), 12)),
                                        int(normal(data.get_n(2), 14)),
                                        int(normal(data.get_n(3), 16)),
                                        int(normal(data.get_n(4), 17))])

            prob_buy_item1 = np.array([binomial(daily_customers[0], data.get_i1_param(1)) / daily_customers[0],
                                       binomial(daily_customers[1], data.get_i1_param(2)) / daily_customers[1],
                                       binomial(daily_customers[2], data.get_i1_param(3)) / daily_customers[2],
                                       binomial(daily_customers[3], data.get_i1_param(4)) / daily_customers[3]])

            # Probability that a customer of a class buys the second item given the first + each promo
            # rows: promo code (0: P0, 1: P1, 2: P2, 3: P3)
            # columns: customer group (0: group1, 1: group2, 2: group3, 3: group4)
            prob_buy_item21 = np.array([  # Promo code P0
                [binomial(daily_customers[0], data.get_i21_p0_param(1)) / daily_customers[0],
                 binomial(daily_customers[1], data.get_i21_p0_param(2)) / daily_customers[1],
                 binomial(daily_customers[2], data.get_i21_p0_param(3)) / daily_customers[2],
                 binomial(daily_customers[3], data.get_i21_p0_param(4)) / daily_customers[3]],
                # Promo code P1
                [binomial(daily_customers[0], data.get_i21_p1_param(1)) / daily_customers[0],
                 binomial(daily_customers[1], data.get_i21_p1_param(2)) / daily_customers[1],
                 binomial(daily_customers[2], data.get_i21_p1_param(3)) / daily_customers[2],
                 binomial(daily_customers[3], data.get_i21_p1_param(4)) / daily_customers[3]],
                # Promo code P2
                [binomial(daily_customers[0], data.get_i21_p2_param(1)) / daily_customers[0],
                 binomial(daily_customers[1], data.get_i21_p2_param(2)) / daily_customers[1],
                 binomial(daily_customers[2], data.get_i21_p2_param(3)) / daily_customers[2],
                 binomial(daily_customers[3], data.get_i21_p2_param(4)) / daily_customers[3]],
                # Promo code P3
                [binomial(daily_customers[0], data.get_i21_p3_param(1)) / daily_customers[0],
                 binomial(daily_customers[1], data.get_i21_p3_param(2)) / daily_customers[1],
                 binomial(daily_customers[2], data.get_i21_p3_param(3)) / daily_customers[2],
                 binomial(daily_customers[3], data.get_i21_p3_param(4)) / daily_customers[3]]
            ])

            dataset.loc[len(dataset)] = [1, 0, 0, 0, 1, 0, 0, 0, prob_buy_item21[0][0]]
            dataset.loc[len(dataset)] = [0, 1, 0, 0, 1, 0, 0, 0, prob_buy_item21[0][1]]
            dataset.loc[len(dataset)] = [0, 0, 1, 0, 1, 0, 0, 0, prob_buy_item21[0][2]]
            dataset.loc[len(dataset)] = [0, 0, 0, 1, 1, 0, 0, 0, prob_buy_item21[0][3]]
            dataset.loc[len(dataset)] = [1, 0, 0, 0, 0, 1, 0, 0, prob_buy_item21[1][0]]
            dataset.loc[len(dataset)] = [0, 1, 0, 0, 0, 1, 0, 0, prob_buy_item21[1][1]]
            dataset.loc[len(dataset)] = [0, 0, 1, 0, 0, 1, 0, 0, prob_buy_item21[1][2]]
            dataset.loc[len(dataset)] = [0, 0, 0, 1, 0, 1, 0, 0, prob_buy_item21[1][3]]
            dataset.loc[len(dataset)] = [1, 0, 0, 0, 0, 0, 1, 0, prob_buy_item21[2][0]]
            dataset.loc[len(dataset)] = [0, 1, 0, 0, 0, 0, 1, 0, prob_buy_item21[2][1]]
            dataset.loc[len(dataset)] = [0, 0, 1, 0, 0, 0, 1, 0, prob_buy_item21[2][2]]
            dataset.loc[len(dataset)] = [0, 0, 0, 1, 0, 0, 1, 0, prob_buy_item21[2][3]]
            dataset.loc[len(dataset)] = [1, 0, 0, 0, 0, 0, 0, 1, prob_buy_item21[3][0]]
            dataset.loc[len(dataset)] = [0, 1, 0, 0, 0, 0, 0, 1, prob_buy_item21[3][1]]
            dataset.loc[len(dataset)] = [0, 0, 1, 0, 0, 0, 0, 1, prob_buy_item21[3][2]]
            dataset.loc[len(dataset)] = [0, 0, 0, 1, 0, 0, 0, 1, prob_buy_item21[3][3]]

            dataset2.loc[len(dataset2)] = [1, 0, 0, 0,prob_buy_item1[0]]
            dataset2.loc[len(dataset2)] = [0, 1, 0, 0,prob_buy_item1[1]]
            dataset2.loc[len(dataset2)] = [0, 0, 1, 0,prob_buy_item1[2]]
            dataset2.loc[len(dataset2)] = [0, 0, 0, 1,prob_buy_item1[3]]

            sum_daily_customer += daily_customers

        target = 'TARGET'
        features = dataset.columns[dataset.columns != target]
        features2 = dataset2.columns[dataset2.columns != target]

        X = dataset[features].values
        y = dataset[target].values

        X2 = dataset2[features2].values
        y2 = dataset2[target].values

        regressor = RandomForestRegressor(n_estimators=100, random_state=1234)
        regressor.fit(X, y)

        regressor2 = RandomForestRegressor(n_estimators=100, random_state=1234)
        regressor2.fit(X2, y2)

        prediction_set = pd.DataFrame(columns=['c1', 'c2', 'c3', 'c4', 'P0', 'P1', 'P2', 'P3'])
        prediction_set.loc[len(prediction_set)] = [1, 0, 0, 0, 1, 0, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 1, 0, 0, 1, 0, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 1, 0, 1, 0, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 0, 1, 1, 0, 0, 0]
        prediction_set.loc[len(prediction_set)] = [1, 0, 0, 0, 0, 1, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 1, 0, 0, 0, 1, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 1, 0, 0, 1, 0, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 0, 1, 0, 1, 0, 0]
        prediction_set.loc[len(prediction_set)] = [1, 0, 0, 0, 0, 0, 1, 0]
        prediction_set.loc[len(prediction_set)] = [0, 1, 0, 0, 0, 0, 1, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 1, 0, 0, 0, 1, 0]
        prediction_set.loc[len(prediction_set)] = [0, 0, 0, 1, 0, 0, 1, 0]
        prediction_set.loc[len(prediction_set)] = [1, 0, 0, 0, 0, 0, 0, 1]
        prediction_set.loc[len(prediction_set)] = [0, 1, 0, 0, 0, 0, 0, 1]
        prediction_set.loc[len(prediction_set)] = [0, 0, 1, 0, 0, 0, 0, 1]
        prediction_set.loc[len(prediction_set)] = [0, 0, 0, 1, 0, 0, 0, 1]

        prediction_set2 = pd.DataFrame(columns=['c1', 'c2', 'c3', 'c4'])
        prediction_set2.loc[len(prediction_set2)] = [1, 0, 0, 0]
        prediction_set2.loc[len(prediction_set2)] = [0, 1, 0, 0]
        prediction_set2.loc[len(prediction_set2)] = [0, 0, 1, 0]
        prediction_set2.loc[len(prediction_set2)] = [0, 0, 0, 1]

        res = regressor.predict(prediction_set.values)
        res2 = regressor2.predict(prediction_set2.values)

        print(res2)

        prob_buy_item21 = np.array([[res[0], res[1], res[2], res[3]],
                                    [res[4], res[5], res[6], res[7]],
                                    [res[8], res[9], res[10], res[11]],
                                    [res[12], res[13], res[14], res[15]]])

        prob_buy_item1 = np.array([res2[0], res2[1], res2[2], res2[3]])
