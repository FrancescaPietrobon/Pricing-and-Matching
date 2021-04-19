from Group import *
from Data import *
from Item import *
from LP_optimization import *
from UCB1 import *
from TS_Learner import *
from Environment import *

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal, binomial

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
        # We choose a conversion rate for item 1 and a margin associated to 6 prices (n_arms):
        # €225 --> conversion rate: 0.65 --> margin: €75
        # €250 --> conversion rate: 0.57 --> margin: €100
        # €275 --> conversion rate: 0.51 --> margin: €125
        # €300 --> conversion rate: 0.43 --> margin: €150
        # €325 --> conversion rate: 0.39 --> margin: €175
        # €350 --> conversion rate: 0.30 --> margin: €200
        # €375 --> conversion rate: 0.27 --> margin: €225

        # We also have:
        # - number of daily customers per class
        # - price of item 2 is fixed
        # - promo code assignment is fixed
        # - conversion rate item 2 given item 1

        # Reward for each price (= arm):
        # €225 --> €75 * 0.65 * n_customers
        # €250 --> €100 * 0.57 * n_customers
        # €275 --> €125 * 0.51 * n_customers
        # €300 --> €150 * 0.43 * n_customers + LP_objective_function_result
        # €325 --> €175 * 0.39 * n_customers
        # €350 --> €200 * 0.30 * n_customers
        # €375 --> €225 * 0.27 * n_customers

        T = 365
        n_arms = np.ceil((T * np.log(T))**(1/4)).astype(int)                    # With T = 365 we have n_arms = 7 (ln)
        p = np.array([0.65, 0.57, 0.51, 0.43, 0.39, 0.30, 0.27])
        opt = p[0]
        env = Environment(n_arms=n_arms, probabilities=p)

        n_experiments = 50
        ucb1_rewards_per_experiment = []
        ts_rewards_per_experiment = []

        for e in range(n_experiments):
            ucb1_learner = UCB1(n_arms=n_arms)
            ts_learner = TS_Learner(n_arms=n_arms)

            for t in range(0, T):
                # UCB1 Learner
                pulled_arm = ucb1_learner.pull_arm()
                reward = env.round(pulled_arm)
                ucb1_learner.update(pulled_arm, reward)

                # Thompson Sampling Learner
                pulled_arm = ts_learner.pull_arm()
                reward = env.round(pulled_arm)
                ts_learner.update(pulled_arm, reward)

            ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
            ts_rewards_per_experiment.append(ts_learner.collected_rewards)

        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), "r")
        plt.plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiment, axis=0)), "b")
        plt.legend(["TS", "UCB1"])
        plt.show()
