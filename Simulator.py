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
        # €225 --> conversion rate: 0.65  --> margin: €75
        # €250 --> conversion rate: 0.57 --> margin: €100
        # €275 --> conversion rate: 0.51 --> margin: €125
        # €300 --> conversion rate: 0.43 --> margin: €150
        # €325 --> conversion rate: 0.39 --> margin: €175
        # €350 --> conversion rate: 0.30 --> margin: €200
        # €375 --> conversion rate: 0.27 --> margin: €225
        conversion_rate = [0.65, 0.57, 0.51, 0.43, 0.39, 0.30, 0.27]
        daily_customer = [380, 220, 267, 124]

        # We also have:
        # - number of daily customers per class
        # - price of item 2 is fixed
        # - promo code assignment is fixed
        # - conversion rate item 2 given item 1

        # Reward for each price (= arm):
        # €225 --> €75 * 0.65 * n_customers + LP (applied for the 0.65 * n_customers )
        # €250 --> €100 * 0.57 * n_customers + LP (applied for the 0.57 * n_customers )
        # €275 --> €125 * 0.51 * n_customers + LP (applied for the 0.51 * n_customers )
        # €300 --> €150 * 0.43 * n_customers + LP (applied for the 0.43 * n_customers )
        # €325 --> €175 * 0.39 * n_customers + LP (applied for the 0.39 * n_customers )
        # €350 --> €200 * 0.30 * n_customers + LP (applied for the 0.30 * n_customers )
        # €375 --> €225 * 0.27 * n_customers + LP (applied for the 0.27 * n_customers )
        margin = [75, 100, 125, 150, 175, 200, 225]
        price = 50
        p1 = 0.1
        p2 = 0.2
        p3 = 0.5
        weights = np.array([[0.92553191, 1, 0, 1],
                            [0, 0, 0.74339623, 0],
                            [0, 0, 0.25660377, 0],
                            [0.07446809, 0, 0, 0]])

        pr_c1_p0 = [0.47, 0.43, 0.38, 0.36, 0.34, 0.31, 0.30]
        pr_c2_p0 = [0.34, 0.29, 0.22, 0.2, 0.19, 0.15, 0.13]
        pr_c3_p0 = [0.40, 0.36, 0.32, 0.29, 0.27, 0.20, 0.17]
        pr_c4_p0 = [0.29, 0.25, 0.21, 0.15, 0.13, 0.11, 0.10]
        pr_c1_p1 = [0.59, 0.53, 0.48, 0.41, 0.39, 0.35, 0.31]
        pr_c2_p1 = [0.39, 0.37, 0.29, 0.26, 0.23, 0.19, 0.17]
        pr_c3_p1 = [0.29, 0.27, 0.26, 0.25, 0.20, 0.12, 0.08]
        pr_c4_p1 = [0.30, 0.25, 0.24, 0.23, 0.19, 0.16, 0.15]
        pr_c1_p2 = [0.62, 0.57, 0.50, 0.44, 0.42, 0.37, 0.30]
        pr_c2_p2 = [0.39, 0.35, 0.31, 0.27, 0.23, 0.21, 0.20]
        pr_c3_p2 = [0.49, 0.45, 0.35, 0.32, 0.29, 0.22, 0.21]
        pr_c4_p2 = [0.30, 0.27, 0.21, 0.19, 0.17, 0.13, 0.10]
        pr_c1_p3 = [0.92, 0.85, 0.79, 0.76, 0.69, 0.58, 0.50]
        pr_c2_p3 = [0.79, 0.73, 0.68, 0.63, 0.56, 0.53, 0.47]
        pr_c3_p3 = [0.80, 0.71, 0.64, 0.61, 0.58, 0.51, 0.43]
        pr_c4_p3 = [0.54, 0.49, 0.47, 0.46, 0.44, 0.35, 0.31]


        T = 2500
        n_arms = 7 # np.ceil((T * np.log(T))**(1/4)).astype(int)                    # With T = 365 we have n_arms = 7 (ln)
        objective = np.zeros(n_arms)
        for i in range(7):
            objective[i] = margin[i]*conversion_rate[i]*sum(daily_customer) + conversion_rate[i]*(
                           daily_customer[0]*price*pr_c1_p0[i]*weights[0][0] + daily_customer[1]*price*pr_c2_p0[i]*weights[0][1]+ daily_customer[2]*price*pr_c3_p0[i]*weights[0][2]+ daily_customer[3]*price*pr_c4_p0[i]*weights[0][3]+
                           daily_customer[0]*price*pr_c1_p1[i]*(1-p1)*weights[1][0]+ daily_customer[1]*price*pr_c2_p1[i]*(1-p1)*weights[1][1]+ daily_customer[2]*price*pr_c3_p1[i]*(1-p1)*weights[1][2]+ daily_customer[3]*price*pr_c4_p1[i]*(1-p1)*weights[1][3]+
                           daily_customer[0]*price*pr_c1_p2[i]*(1-p2)*weights[2][0]+ daily_customer[1]*price*pr_c2_p2[i]*(1-p2)*weights[2][1]+ daily_customer[2]*price*pr_c3_p2[i]*(1-p2)*weights[2][2]+ daily_customer[3]*price*pr_c4_p2[i]*(1-p2)*weights[2][3]+
                           daily_customer[0]*price*pr_c1_p3[i]*(1-p3)*weights[3][0]+ daily_customer[1]*price*pr_c2_p3[i]*(1-p3)*weights[3][1]+ daily_customer[2]*price*pr_c3_p3[i]*(1-p3)*weights[3][2]+ daily_customer[3]*price*pr_c4_p3[i]*(1-p3)*weights[3][3])

        objective = objective / np.linalg.norm(objective)
        opt = max(objective)
        env = Environment(n_arms=n_arms, objective=objective)

        n_experiments = 100
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
        plt.legend(["TS", "UCB1"], title= "STEP 3")
        plt.show()

        plt.figure(1)
        plt.xlabel("t")
        plt.ylabel("Reward")
        plt.plot(np.mean(ts_rewards_per_experiment, axis=0), "r")
        plt.plot(np.mean(ucb1_rewards_per_experiment, axis=0), "b")
        plt.legend(["TS", "UCB1"], title="STEP 3")
        plt.show()

########################################################################################################################


    def simulation_step_4(self):
        # We choose a conversion rate for item 1 and a margin associated to 6 prices (n_arms):
        # €225 --> conversion rate: 0.65  --> margin: €75
        # €250 --> conversion rate: 0.57 --> margin: €100
        # €275 --> conversion rate: 0.51 --> margin: €125
        # €300 --> conversion rate: 0.43 --> margin: €150
        # €325 --> conversion rate: 0.39 --> margin: €175
        # €350 --> conversion rate: 0.30 --> margin: €200
        # €375 --> conversion rate: 0.27 --> margin: €225
        conversion_rate = [0.65, 0.57, 0.51, 0.43, 0.39, 0.30, 0.27]
        daily_customer = [380, 220, 267, 124]
        daily_promos = [693, 198, 69, 29]

        # We also have:
        # - number of daily customers per class
        # - price of item 2 is fixed
        # - promo code assignment is fixed
        # - conversion rate item 2 given item 1

        # Reward for each price (= arm):
        # €225 --> €75 * 0.65 * n_customers + LP (applied for the 0.65 * n_customers )
        # €250 --> €100 * 0.57 * n_customers + LP (applied for the 0.57 * n_customers )
        # €275 --> €125 * 0.51 * n_customers + LP (applied for the 0.51 * n_customers )
        # €300 --> €150 * 0.43 * n_customers + LP (applied for the 0.43 * n_customers )
        # €325 --> €175 * 0.39 * n_customers + LP (applied for the 0.39 * n_customers )
        # €350 --> €200 * 0.30 * n_customers + LP (applied for the 0.30 * n_customers )
        # €375 --> €225 * 0.27 * n_customers + LP (applied for the 0.27 * n_customers )
        margin = [75, 100, 125, 150, 175, 200, 225]


        T = 2500
        n_arms = 7  # np.ceil((T * np.log(T))**(1/4)).astype(int)                    # With T = 365 we have n_arms = 7 (ln)
        objective = np.zeros(n_arms)
        for i in range(7):
            objective[i] = margin[i] * conversion_rate[i]

        objective = objective / np.linalg.norm(objective)
        opt = max(objective)
        env = Environment(n_arms=n_arms, objective=objective)

        n_experiments = 100
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
        plt.legend(["TS", "UCB1"], title = "STEP 4")
        plt.show()

        plt.figure(1)
        plt.xlabel("t")
        plt.ylabel("Reward")
        plt.plot(np.mean(ts_rewards_per_experiment, axis=0), "r")
        plt.plot(np.mean(ucb1_rewards_per_experiment, axis=0), "b")
        plt.legend(["TS", "UCB1"], title="STEP 4")
        plt.show()