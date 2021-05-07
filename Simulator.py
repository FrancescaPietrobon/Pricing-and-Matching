import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal, binomial
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

from Data import *
from Environment import *
from Group import *
from Item import *
from LP_optimization import *
from Step3.TS_Learner_item1 import *
from Step3.UCB1_item1 import *
from Step4.UCB1_item2 import *
from Step4.UCB1_item1_new import *
from Step4.TS_Learner_item1_new import *
from TS_Learner import *
from UCB1 import *
from UCB_matching import *

np.random.seed(1234)


class Simulator:
    def __init__(self):
        self.customers_groups = np.array([Group(1), Group(2), Group(3), Group(4)])
        self.item1 = Item("Apple Watch", 300, 88)
        self.item2 = Item("Personalized wristband", 50, 16)
        self.discount_p1 = 0.1
        self.discount_p2 = 0.2
        self.discount_p3 = 0.5
        self.data = Data()

    def simulation_step_1(self, p0_frac, p1_frac, p2_frac, p3_frac):

        # Daily number of customers per class = Gaussian TODO: are sigmas correct?
        daily_customers = self.data.get_daily_customers()

        # Number of promo codes available daily (fixed fraction of the daily number of customers)
        daily_promos = [int(sum(daily_customers) * p0_frac),
                        int(sum(daily_customers) * p1_frac),
                        int(sum(daily_customers) * p2_frac),
                        int(sum(daily_customers) * p3_frac)]

        # Probability that a customer of a class buys the second item given the first + each promo
        # rows: promo code (0: P0, 1: P1, 2: P2, 3: P3)
        # columns: customer group (0: group1, 1: group2, 2: group3, 3: group4)
        prob_buy_item21 = self.data.get_conversion_rates_item21()

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

        # Conversion rates for item 1 (one per arm)
        conversion_rates_item1 = np.array([[0.9, 0.84, 0.72, 0.59, 0.50, 0.42, 0.23, 0.13, 0.07],
                                          [0.87, 0.75, 0.57, 0.44, 0.36, 0.29, 0.13, 0.10, 0.02],
                                          [0.89, 0.78, 0.62, 0.48, 0.45, 0.36, 0.17, 0.12, 0.05],
                                          [0.88, 0.78, 0.59, 0.44, 0.37, 0.31, 0.15, 0.13, 0.03]])

        # Number of daily customers (one per class) - Taken by step 1
        daily_customers = np.array([380, 220, 267, 124])

        # Promo assigment (row: promo; column: customer class) - Taken by step 1
        # TODO use also the matrix of the second experiment of Step 1 (give matrix in input)
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
            objective[i] = (daily_customers[0] * conversion_rates_item1[0][i] +
                            daily_customers[1] * conversion_rates_item1[1][i] +
                            daily_customers[2] * conversion_rates_item1[2][i] +
                            daily_customers[3] * conversion_rates_item1[3][i]) * prices[i] + self.item2.get_price() * (
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[0][0][i] * weights[0][0] +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[0][1][i] * weights[0][1] +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[0][2][i] * weights[0][2] +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[0][3][i] * weights[0][3] +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[1][0][i] * weights[1][0] * (1-self.discount_p1) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[1][1][i] * weights[1][1] * (1-self.discount_p1) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[1][2][i] * weights[1][2] * (1-self.discount_p1) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[1][3][i] * weights[1][3] * (1-self.discount_p1) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[2][0][i] * weights[2][0] * (1-self.discount_p2) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[2][1][i] * weights[2][1] * (1-self.discount_p2) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[2][2][i] * weights[2][2] * (1-self.discount_p2) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[2][3][i] * weights[2][3] * (1-self.discount_p2) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[3][0][i] * weights[3][0] * (1-self.discount_p3) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[3][1][i] * weights[3][1] * (1-self.discount_p3) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[3][2][i] * weights[3][2] * (1-self.discount_p3) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[3][3][i] * weights[3][3] * (1-self.discount_p3))

        reward_item2 = np.zeros([4, n_arms])
        for i in range(n_arms):
            reward_item2[0, i] = self.item2.get_price() * (
                                    conversion_rates_item21[0, 0, i] * weights[0][0] +
                                    conversion_rates_item21[1, 0, i] * weights[1][0] * (1 - self.discount_p1) +
                                    conversion_rates_item21[2, 0, i] * weights[2][0] * (1 - self.discount_p2) +
                                    conversion_rates_item21[3, 0, i] * weights[3][0] * (1 - self.discount_p3))
            reward_item2[1, i] = self.item2.get_price() * (
                                    conversion_rates_item21[0, 1, i] * weights[0][1] +
                                    conversion_rates_item21[1, 1, i] * weights[1][1] * (1 - self.discount_p1) +
                                    conversion_rates_item21[2, 1, i] * weights[2][1] * (1 - self.discount_p2) +
                                    conversion_rates_item21[3, 1, i] * weights[3][1] * (1 - self.discount_p3))
            reward_item2[2, i] = self.item2.get_price() * (
                                    conversion_rates_item21[0, 2, i] * weights[0][2] +
                                    conversion_rates_item21[1, 2, i] * weights[1][2] * (1 - self.discount_p1) +
                                    conversion_rates_item21[2, 2, i] * weights[2][2] * (1 - self.discount_p2) +
                                    conversion_rates_item21[3, 2, i] * weights[3][2] * (1 - self.discount_p3))
            reward_item2[3, i] = self.item2.get_price() * (
                                    conversion_rates_item21[0, 3, i] * weights[0][3] +
                                    conversion_rates_item21[1, 3, i] * weights[1][3] * (1 - self.discount_p1) +
                                    conversion_rates_item21[2, 3, i] * weights[2][3] * (1 - self.discount_p2) +
                                    conversion_rates_item21[3, 3, i] * weights[3][3] * (1 - self.discount_p3))

        # Storing the optimal objective value to compute the regret later
        opt = max(objective)

        # Launching the experiments, using both UCB1 and Thompson Sampling
        n_experiments = 100
        ucb1_rewards_per_experiment = []
        ts_rewards_per_experiment = []

        for e in range(n_experiments):
            print(e+1)
            env = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)
            ucb1_learner = UCB1_item1(n_arms=n_arms, daily_customers=daily_customers, prices=prices, reward_item2=reward_item2)
            ts_learner = TS_Learner_item1(n_arms=n_arms, daily_customers=daily_customers, prices=prices, reward_item2=reward_item2)

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

        # Plotting the regret and the reward
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), "r")
        plt.plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiment, axis=0)), "b")
        plt.legend(["TS", "UCB1"], title="STEP 3")
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


        # Time horizon
        T = 1000

        # Number of arms (computed as np.ceil((T * np.log(T))**(1/4)).astype(int))
        n_arms = 9

        # Candidate prices (one per arm) - The central one (€300) is taken by step 1
        prices = np.array([50, 100, 150, 200, 300, 400, 450, 500, 550])

        # Conversion rates for item 1 (one per arm)
        conversion_rates_item1 = np.array([[0.9, 0.84, 0.72, 0.59, 0.50, 0.42, 0.23, 0.13, 0.07],
                                           [0.87, 0.75, 0.57, 0.44, 0.36, 0.29, 0.13, 0.10, 0.02],
                                           [0.89, 0.78, 0.62, 0.48, 0.45, 0.36, 0.17, 0.12, 0.05],
                                           [0.88, 0.78, 0.59, 0.44, 0.37, 0.31, 0.15, 0.13, 0.03]])

        # Number of daily customers (one per class) - Taken by step 1
        # TODO set to 1 since we don't care (keep or remove?)
        daily_customers = np.ones(4)

        # Promo assigment (row: promo; column: customer class) - Taken by step 1
        # TODO use also the matrix of the second experiment of Step 1 (give matrix in input)
        weights = np.array([[0.92553191, 1, 0, 1],
                            [0, 0, 0.74339623, 0],
                            [0, 0, 0.25660377, 0],
                            [0.07446809, 0, 0, 0]])

        # Conversion rates for item 2 given item 1 and promo (one row per class per promo; one element per arm)
        conversion_rates_item21 = self.data.get_conversion_rates_item21()

        # Computing the objective array (one element per arm)
        objective = np.zeros(n_arms)
        for i in range(n_arms):
            objective[i] = (daily_customers[0] * conversion_rates_item1[0][i] +
                            daily_customers[1] * conversion_rates_item1[1][i] +
                            daily_customers[2] * conversion_rates_item1[2][i] +
                            daily_customers[3] * conversion_rates_item1[3][i]) * prices[i] + self.item2.get_price() * (
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[0][0] * weights[0][0] +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[0][1] * weights[0][1] +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[0][2] * weights[0][2] +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[0][3] * weights[0][3] +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[1][0] * weights[1][0] * (1 - self.discount_p1) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[1][1] * weights[1][1] * (1 - self.discount_p1) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[1][2] * weights[1][2] * (1 - self.discount_p1) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[1][3] * weights[1][3] * (1 - self.discount_p1) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[2][0] * weights[2][0] * (1 - self.discount_p2) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[2][1] * weights[2][1] * (1 - self.discount_p2) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[2][2] * weights[2][2] * (1 - self.discount_p2) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[2][3] * weights[2][3] * (1 - self.discount_p2) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[3][0] * weights[3][0] * (1 - self.discount_p3) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[3][1] * weights[3][1] * (1 - self.discount_p3) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[3][2] * weights[3][2] * (1 - self.discount_p3) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[3][3] * weights[3][3] * (1 - self.discount_p3))

        # Storing the optimal objective value to compute the regret later
        opt = max(objective)

        # Launching the experiments, using both UCB1 and Thompson Sampling
        n_experiments = 100
        ucb1_rewards_per_experiment_item1 = []
        ts_rewards_per_experiment_item1 = []

        for e in range(n_experiments):
            print(e + 1)

            env_item2_class1 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 0])
            env_item2_class2 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 1])
            env_item2_class3 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 2])
            env_item2_class4 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 3])

            ucb1_learner_item2_class1 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class2 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class3 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class4 = UCB1_item2(n_arms=4)

            env_item1 = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)

            ucb1_learner_item1 = UCB1_item1_new(n_arms=n_arms, daily_customers=daily_customers, prices=prices, reward_item2=np.zeros(4))
            ts_learner_item1 = TS_Learner_item1_new(n_arms=n_arms, daily_customers=daily_customers, prices=prices, reward_item2=np.zeros(4))

            for t in range(0, T):
                # Item 2 Class 1
                pulled_arm = ucb1_learner_item2_class1.pull_arm()
                reward = env_item2_class1.round(pulled_arm)
                ucb1_learner_item2_class1.update(pulled_arm, reward)

                # Item 2 CLass 2
                pulled_arm = ucb1_learner_item2_class2.pull_arm()
                reward = env_item2_class2.round(pulled_arm)
                ucb1_learner_item2_class2.update(pulled_arm, reward)

                # Item 2 Class 3
                pulled_arm = ucb1_learner_item2_class3.pull_arm()
                reward = env_item2_class3.round(pulled_arm)
                ucb1_learner_item2_class3.update(pulled_arm, reward)

                # Item 2 Class 4
                pulled_arm = ucb1_learner_item2_class4.pull_arm()
                reward = env_item2_class4.round(pulled_arm)
                ucb1_learner_item2_class4.update(pulled_arm, reward)

                # Item 1
                if t >= 4:
                    conversion_rates_item2_ub = np.zeros([4, 4])
                    conversion_rates_item2_ub[:, 0] = ucb1_learner_item2_class1.get_empirical_means()
                    conversion_rates_item2_ub[:, 1] = ucb1_learner_item2_class2.get_empirical_means()
                    conversion_rates_item2_ub[:, 2] = ucb1_learner_item2_class3.get_empirical_means()
                    conversion_rates_item2_ub[:, 3] = ucb1_learner_item2_class4.get_empirical_means()

                    reward_item2 = np.zeros(4)
                    reward_item2[0] = self.item2.get_price() * (
                            conversion_rates_item2_ub[0, 0] * weights[0][0] +
                            conversion_rates_item2_ub[1, 0] * weights[1][0] * (1 - self.discount_p1) +
                            conversion_rates_item2_ub[2, 0] * weights[2][0] * (1 - self.discount_p2) +
                            conversion_rates_item2_ub[3, 0] * weights[3][0] * (1 - self.discount_p3))
                    reward_item2[1] = self.item2.get_price() * (
                            conversion_rates_item2_ub[0, 1] * weights[0][1] +
                            conversion_rates_item2_ub[1, 1] * weights[1][1] * (1 - self.discount_p1) +
                            conversion_rates_item2_ub[2, 1] * weights[2][1] * (1 - self.discount_p2) +
                            conversion_rates_item2_ub[3, 1] * weights[3][1] * (1 - self.discount_p3))
                    reward_item2[2] = self.item2.get_price() * (
                            conversion_rates_item2_ub[0, 2] * weights[0][2] +
                            conversion_rates_item2_ub[1, 2] * weights[1][2] * (1 - self.discount_p1) +
                            conversion_rates_item2_ub[2, 2] * weights[2][2] * (1 - self.discount_p2) +
                            conversion_rates_item2_ub[3, 2] * weights[3][2] * (1 - self.discount_p3))
                    reward_item2[3] = self.item2.get_price() * (
                            conversion_rates_item2_ub[0, 3] * weights[0][3] +
                            conversion_rates_item2_ub[1, 3] * weights[1][3] * (1 - self.discount_p1) +
                            conversion_rates_item2_ub[2, 3] * weights[2][3] * (1 - self.discount_p2) +
                            conversion_rates_item2_ub[3, 3] * weights[3][3] * (1 - self.discount_p3))

                    ucb1_learner_item1.update_reward_item2(reward_item2)
                    ts_learner_item1.update_reward_item2(reward_item2)

                    pulled_arm = ucb1_learner_item1.pull_arm()
                    reward = env_item1.round(pulled_arm)
                    ucb1_learner_item1.update(pulled_arm, reward)

                    pulled_arm = ts_learner_item1.pull_arm()
                    reward = env_item1.round(pulled_arm)
                    ts_learner_item1.update(pulled_arm, reward)

            ucb1_rewards_per_experiment_item1.append(ucb1_learner_item1.collected_rewards)
            ts_rewards_per_experiment_item1.append(ts_learner_item1.collected_rewards)

        # Plotting the regret and the reward
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment_item1, axis=0)), "r")
        plt.plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiment_item1, axis=0)), "b")
        plt.legend(["TS", "UCB1"], title="STEP 4")
        plt.show()

        plt.figure(1)
        plt.xlabel("t")
        plt.ylabel("Reward")
        plt.plot(np.mean(ts_rewards_per_experiment_item1, axis=0), "r")
        plt.plot(np.mean(ucb1_rewards_per_experiment_item1, axis=0), "b")
        plt.legend(["TS", "UCB1"], title="STEP 4")
        plt.show()

########################################################################################################################

    def simulation_step_5(self, p0_frac, p1_frac, p2_frac, p3_frac):

        conversion_rates_item21 = self.data.get_conversion_rates_item21()


        daily_customers = np.ones(4)

        discounts = np.array([1-self.discount_p1, 1-self.discount_p2, 1-self.discount_p3])

        p_frac = np.array([p1_frac, p2_frac, p3_frac])

        # Assumption:
        # The number of promos of every type of promo is smaller than the number of customers of every class
        objective = np.zeros((3, 4))
        objective[0][0] = self.item2.get_price() * (1 - self.discount_p1) * conversion_rates_item21[1][0] * daily_customers[0] * p1_frac
        objective[0][1] = self.item2.get_price() * (1 - self.discount_p1) * conversion_rates_item21[1][1] * daily_customers[1] * p1_frac
        objective[0][2] = self.item2.get_price() * (1 - self.discount_p1) * conversion_rates_item21[1][2] * daily_customers[2] * p1_frac
        objective[0][3] = self.item2.get_price() * (1 - self.discount_p1) * conversion_rates_item21[1][3] * daily_customers[3] * p1_frac
        objective[1][0] = self.item2.get_price() * (1 - self.discount_p2) * conversion_rates_item21[2][0] * daily_customers[0] * p2_frac
        objective[1][1] = self.item2.get_price() * (1 - self.discount_p2) * conversion_rates_item21[2][1] * daily_customers[1] * p2_frac
        objective[1][2] = self.item2.get_price() * (1 - self.discount_p2) * conversion_rates_item21[2][2] * daily_customers[2] * p2_frac
        objective[1][3] = self.item2.get_price() * (1 - self.discount_p2) * conversion_rates_item21[2][3] * daily_customers[3] * p2_frac
        objective[2][0] = self.item2.get_price() * (1 - self.discount_p3) * conversion_rates_item21[3][0] * daily_customers[0] * p3_frac
        objective[2][1] = self.item2.get_price() * (1 - self.discount_p3) * conversion_rates_item21[3][1] * daily_customers[1] * p3_frac
        objective[2][2] = self.item2.get_price() * (1 - self.discount_p3) * conversion_rates_item21[3][2] * daily_customers[2] * p3_frac
        objective[2][3] = self.item2.get_price() * (1 - self.discount_p3) * conversion_rates_item21[3][3] * daily_customers[3] * p3_frac

        probabilities = np.zeros((3, 4))

        opt = linear_sum_assignment(-objective)

        n_exp = 100
        T = 5000
        regret_ucb = np.zeros((n_exp, T))
        reward_ucb = []
        for e in range(n_exp):

            print(e + 1)

            env_item2_class1 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 0])
            env_item2_class2 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 1])
            env_item2_class3 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 2])
            env_item2_class4 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 3])

            ucb1_learner_item2_class1 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class2 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class3 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class4 = UCB1_item2(n_arms=4)

            env = Environment_First(probabilities.size, probabilities)
            learner = UCB_Matching(probabilities.size, *probabilities.shape, self.item2.get_price(), discounts, p_frac)

            rew_UCB = []
            opt_rew = []

            for t in range(T):

                # Item 2 Class 1
                pulled_arm = ucb1_learner_item2_class1.pull_arm()
                reward = env_item2_class1.round(pulled_arm)
                ucb1_learner_item2_class1.update(pulled_arm, reward)

                # Item 2 CLass 2
                pulled_arm = ucb1_learner_item2_class2.pull_arm()
                reward = env_item2_class2.round(pulled_arm)
                ucb1_learner_item2_class2.update(pulled_arm, reward)

                # Item 2 Class 3
                pulled_arm = ucb1_learner_item2_class3.pull_arm()
                reward = env_item2_class3.round(pulled_arm)
                ucb1_learner_item2_class3.update(pulled_arm, reward)

                # Item 2 Class 4
                pulled_arm = ucb1_learner_item2_class4.pull_arm()
                reward = env_item2_class4.round(pulled_arm)
                ucb1_learner_item2_class4.update(pulled_arm, reward)

                #Matching

                conversion_rates_item2_ub = np.zeros([4, 4])
                conversion_rates_item2_ub[:, 0] = ucb1_learner_item2_class1.get_empirical_means()
                conversion_rates_item2_ub[:, 1] = ucb1_learner_item2_class2.get_empirical_means()
                conversion_rates_item2_ub[:, 2] = ucb1_learner_item2_class3.get_empirical_means()
                conversion_rates_item2_ub[:, 3] = ucb1_learner_item2_class4.get_empirical_means()

                probabilities = np.zeros((3,4))
                for i in range(3):
                    for j in range(4):
                        probabilities[i][j] = conversion_rates_item2_ub[i+1,j]

                env.set_probabilities(probabilities)

                pulled_arms = learner.pull_arm()
                rewards = env.round(pulled_arms)
                learner.update(pulled_arms, rewards)

                rew_UCB.append((rewards * self.item2.get_price() * discounts * p_frac).sum())
                opt_rew.append(objective[opt].sum())

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

    def simulation_step_6(self, p0_frac, p1_frac, p2_frac, p3_frac):
        # Learning the matching
        conversion_rates_item21 = self.data.get_conversion_rates_item21()

        conversion_rates_item21_by_price = np.array([[self.data.get_conversion_rates_item21() + 0.1],
                                                     [self.data.get_conversion_rates_item21()],
                                                     [self.data.get_conversion_rates_item21() - 0.1]])

        daily_customers = np.ones(4)

        discounts = np.array([1 - self.discount_p1, 1 - self.discount_p2, 1 - self.discount_p3])

        p_frac = np.array([p1_frac, p2_frac, p3_frac])

        n_arms = 9

        # Candidate prices (one per arm) - The central one (€300) is taken by step 1
        prices = np.array([50, 100, 150, 200, 300, 400, 450, 500, 550])

        prices_item2 = np.array([40, 50, 60])

        # Conversion rates for item 1 (one per arm)
        conversion_rates_item1 = np.array([[0.9, 0.84, 0.72, 0.59, 0.50, 0.42, 0.23, 0.13, 0.07],
                                           [0.87, 0.75, 0.57, 0.44, 0.36, 0.29, 0.13, 0.10, 0.02],
                                           [0.89, 0.78, 0.62, 0.48, 0.45, 0.36, 0.17, 0.12, 0.05],
                                           [0.88, 0.78, 0.59, 0.44, 0.37, 0.31, 0.15, 0.13, 0.03]])

        weights = np.array([[0.92553191, 1, 0, 1],
                            [0, 0, 0.74339623, 0],
                            [0, 0, 0.25660377, 0],
                            [0.07446809, 0, 0, 0]])

        # Computing the objective array (one element per arm)
        objective = np.zeros(n_arms)
        for i in range(n_arms):
            objective[i] = (daily_customers[0] * conversion_rates_item1[0][i] +
                            daily_customers[1] * conversion_rates_item1[1][i] +
                            daily_customers[2] * conversion_rates_item1[2][i] +
                            daily_customers[3] * conversion_rates_item1[3][i]) * prices[i] + self.item2.get_price() * (
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[0][0] * weights[0][0] +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[0][1] * weights[0][1] +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[0][2] * weights[0][2] +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[0][3] * weights[0][3] +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[1][0] * weights[1][0] * (1 - self.discount_p1) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[1][1] * weights[1][1] * (1 - self.discount_p1) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[1][2] * weights[1][2] * (1 - self.discount_p1) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[1][3] * weights[1][3] * (1 - self.discount_p1) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[2][0] * weights[2][0] * (1 - self.discount_p2) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[2][1] * weights[2][1] * (1 - self.discount_p2) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[2][2] * weights[2][2] * (1 - self.discount_p2) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[2][3] * weights[2][3] * (1 - self.discount_p2) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[3][0] * weights[3][0] * (1 - self.discount_p3) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[3][1] * weights[3][1] * (1 - self.discount_p3) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[3][2] * weights[3][2] * (1 - self.discount_p3) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[3][3] * weights[3][3] * (1 - self.discount_p3))

        probabilities = np.zeros((3, 4))

        # Storing the optimal objective value to compute the regret later
        opt = max(objective)

        n_exp = 20
        T = 5000

        ucb1_rewards_per_experiment_item1 = []
        ts_rewards_per_experiment_item1 = []

        for e in range(n_exp):
            print(e + 1)

            env_item2_class1 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, 0].flatten())
            env_item2_class2 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, 1].flatten())
            env_item2_class3 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, 2].flatten())
            env_item2_class4 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, 3].flatten())

            ucb1_learner_item2_class1 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class2 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class3 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class4 = UCB1_item2(n_arms=12)

            ucb1_learner_matching = UCB_Matching(probabilities.size, *probabilities.shape, price=0, discounts=discounts, p_frac=p_frac)
            env_matching = Environment_First(probabilities.size, probabilities)

            env_item1 = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)

            ucb1_learner_item1 = UCB1_item1_new(n_arms=n_arms, daily_customers=daily_customers, prices=prices, reward_item2=np.zeros(4))
            ts_learner_item1 = TS_Learner_item1_new(n_arms=n_arms, daily_customers=daily_customers, prices=prices, reward_item2=np.zeros(4))


            for t in range(T):
                # TODO think about the update
                # Item 2 Class 1
                pulled_arm = ucb1_learner_item2_class1.pull_arm()
                reward = env_item2_class1.round(pulled_arm)
                ucb1_learner_item2_class1.update(pulled_arm, reward)

                # Item 2 CLass 2
                pulled_arm = ucb1_learner_item2_class2.pull_arm()
                reward = env_item2_class2.round(pulled_arm)
                ucb1_learner_item2_class2.update(pulled_arm, reward)

                # Item 2 Class 3
                pulled_arm = ucb1_learner_item2_class3.pull_arm()
                reward = env_item2_class3.round(pulled_arm)
                ucb1_learner_item2_class3.update(pulled_arm, reward)

                # Item 2 Class 4
                pulled_arm = ucb1_learner_item2_class4.pull_arm()
                reward = env_item2_class4.round(pulled_arm)
                ucb1_learner_item2_class4.update(pulled_arm, reward)

                # Matching
                conversion_rates_item2_ub = np.zeros([4, 4])
                argmax_index_item2_class1 = np.argmax([sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[0]) * prices_item2[0],
                                                      sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[1]) * prices_item2[1],
                                                      sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[2]) * prices_item2[2]])
                argmax_index_item2_class2 = np.argmax([sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[0]) * prices_item2[0],
                                                      sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[1]) * prices_item2[1],
                                                      sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[2]) * prices_item2[2]])
                argmax_index_item2_class3 = np.argmax([sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[0]) * prices_item2[0],
                                                      sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[1]) * prices_item2[1],
                                                      sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[2]) * prices_item2[2]])
                argmax_index_item2_class4 = np.argmax([sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[0]) * prices_item2[0],
                                                      sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[1]) * prices_item2[1],
                                                      sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[2]) * prices_item2[2]])

                majority_voting = stats.mode([argmax_index_item2_class1, argmax_index_item2_class2, argmax_index_item2_class3, argmax_index_item2_class4])[0][0]

                conversion_rates_item2_ub[:, 0] = ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[majority_voting]
                conversion_rates_item2_ub[:, 1] = ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[majority_voting]
                conversion_rates_item2_ub[:, 2] = ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[majority_voting]
                conversion_rates_item2_ub[:, 3] = ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[majority_voting]

                probabilities = np.zeros((3, 4))
                for i in range(3):
                    for j in range(4):
                        probabilities[i][j] = conversion_rates_item2_ub[i + 1, j]

                env_matching.set_probabilities(probabilities)
                ucb1_learner_matching.set_price(prices_item2[majority_voting])

                pulled_arms = ucb1_learner_matching.pull_arm()
                rewards = env_matching.round(pulled_arms)
                ucb1_learner_matching.update(pulled_arms, rewards)

                # Price item 1
                res = np.zeros((3, 4))

                res[pulled_arms[0][0]][pulled_arms[1][0]] = 1
                res[pulled_arms[0][1]][pulled_arms[1][1]] = 1
                res[pulled_arms[0][2]][pulled_arms[1][2]] = 1

                p_frac = np.array([p0_frac, p1_frac, p2_frac, p3_frac])
                weights = np.zeros((4, 4))
                for i in range(0, 3):
                    for j in range(0, 4):
                        if res[i][j] == 1:
                            weights[i + 1][j] = p_frac[i + 1] * sum(daily_customers)

                for j in range(0, 4):
                    weights[0][j] = daily_customers[j] - sum(weights[:, j])

                reward_item2 = np.zeros(4)
                reward_item2[0] = prices_item2[majority_voting] * (
                        conversion_rates_item2_ub[0, 0] * weights[0][0] +
                        conversion_rates_item2_ub[1, 0] * weights[1][0] * (1 - self.discount_p1) +
                        conversion_rates_item2_ub[2, 0] * weights[2][0] * (1 - self.discount_p2) +
                        conversion_rates_item2_ub[3, 0] * weights[3][0] * (1 - self.discount_p3))
                reward_item2[1] = prices_item2[majority_voting] * (
                        conversion_rates_item2_ub[0, 1] * weights[0][1] +
                        conversion_rates_item2_ub[1, 1] * weights[1][1] * (1 - self.discount_p1) +
                        conversion_rates_item2_ub[2, 1] * weights[2][1] * (1 - self.discount_p2) +
                        conversion_rates_item2_ub[3, 1] * weights[3][1] * (1 - self.discount_p3))
                reward_item2[2] = prices_item2[majority_voting] * (
                        conversion_rates_item2_ub[0, 2] * weights[0][2] +
                        conversion_rates_item2_ub[1, 2] * weights[1][2] * (1 - self.discount_p1) +
                        conversion_rates_item2_ub[2, 2] * weights[2][2] * (1 - self.discount_p2) +
                        conversion_rates_item2_ub[3, 2] * weights[3][2] * (1 - self.discount_p3))
                reward_item2[3] = prices_item2[majority_voting] * (
                        conversion_rates_item2_ub[0, 3] * weights[0][3] +
                        conversion_rates_item2_ub[1, 3] * weights[1][3] * (1 - self.discount_p1) +
                        conversion_rates_item2_ub[2, 3] * weights[2][3] * (1 - self.discount_p2) +
                        conversion_rates_item2_ub[3, 3] * weights[3][3] * (1 - self.discount_p3))

                ucb1_learner_item1.update_reward_item2(reward_item2)
                ts_learner_item1.update_reward_item2(reward_item2)

                pulled_arm = ucb1_learner_item1.pull_arm()
                reward = env_item1.round(pulled_arm)
                ucb1_learner_item1.update(pulled_arm, reward)

                pulled_arm = ts_learner_item1.pull_arm()
                reward = env_item1.round(pulled_arm)
                ts_learner_item1.update(pulled_arm, reward)

            ucb1_rewards_per_experiment_item1.append(ucb1_learner_item1.collected_rewards)
            ts_rewards_per_experiment_item1.append(ts_learner_item1.collected_rewards)

        # Plotting the regret and the reward
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment_item1, axis=0)), "r")
        plt.plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiment_item1, axis=0)), "b")
        plt.legend(["TS", "UCB1"], title="STEP 6")
        plt.show()

        plt.figure(1)
        plt.xlabel("t")
        plt.ylabel("Reward")
        plt.plot(np.mean(ts_rewards_per_experiment_item1, axis=0), "r")
        plt.plot(np.mean(ucb1_rewards_per_experiment_item1, axis=0), "b")
        plt.legend(["TS", "UCB1"], title="STEP 6")
        plt.show()
