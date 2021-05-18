import matplotlib.pyplot as plt
from numpy.random import normal
from scipy import stats

from Data import *
from Environment import *
from Item import *
from LP_optimization import *
from Learner_Customers import *
from SWTS_Learner import *
from SWTS_Learner_item2 import *
from UCB1_item1 import *
from UCB1_item2 import *
from UCB_matching import *

np.random.seed(1234)


class Simulator:
    def __init__(self):
        self.item1 = Item(name="Apple Watch", price=300, margin=88)
        self.item2 = Item(name="Personalized wristband", price=50, margin=16)
        self.discounts = np.array([0, 0.1, 0.2, 0.5])
        self.data = Data()

    def simulation_step_1(self, promo_fractions):
        # Daily number of customers per class
        daily_customers = self.data.get_daily_customers()

        # Number of promo codes available daily (fixed fraction of the daily number of customers)
        daily_promos = [int(sum(daily_customers) * promo_fractions[0]),
                        int(sum(daily_customers) * promo_fractions[1]),
                        int(sum(daily_customers) * promo_fractions[2]),
                        int(sum(daily_customers) * promo_fractions[3])]

        # Probability that a customer of a class buys the second item given the first + each promo
        # rows: promo (0: P0, 1: P1, 2: P2, 3: P3); columns: customer class (0: class1, 1: class2, 2: class3, 3: class4)
        prob_buy_item21 = self.data.get_conversion_rates_item21()

        # Linear optimization algorithm to find the best matching
        return LP(self.item2.get_price(), self.discounts, prob_buy_item21, daily_promos, daily_customers)

########################################################################################################################

    def simulation_step_3(self):
        # Number of arms (computed as np.ceil((T * np.log(T))**(1/4)).astype(int))
        n_arms = 9

        # Candidate prices (one per arm)
        prices = np.array([50, 100, 150, 200, 300, 400, 450, 500, 550])

        # Conversion rates for item 1 (one per arm)
        conversion_rates_item1 = np.array([[0.9, 0.84, 0.72, 0.59, 0.50, 0.42, 0.23, 0.13, 0.07],
                                          [0.87, 0.75, 0.57, 0.44, 0.36, 0.29, 0.13, 0.10, 0.02],
                                          [0.89, 0.78, 0.62, 0.48, 0.45, 0.36, 0.17, 0.12, 0.05],
                                          [0.88, 0.78, 0.59, 0.44, 0.37, 0.31, 0.15, 0.13, 0.03]])

        # Conversion rates for item 2 given item 1 and promo (row: promo; column: customer class)
        conversion_rates_item2 = self.data.get_conversion_rates_item21()

        # Number of daily customers (one per class)
        daily_customers = self.data.get_daily_customers()

        # Promo assigment (row: promo; column: customer class) - Taken by step 1
        # TODO use also the matrix of the second experiment of Step 1 (give matrix in input)
        weights = np.array([[0.92553191, 1, 0, 1],
                            [0, 0, 0.74339623, 0],
                            [0, 0, 0.25660377, 0],
                            [0.07446809, 0, 0, 0]])

        # Objective array (one element per arm)
        objective = np.zeros(n_arms)
        for i in range(n_arms):
            objective[i] = (daily_customers[0] * conversion_rates_item1[0][i] +
                            daily_customers[1] * conversion_rates_item1[1][i] +
                            daily_customers[2] * conversion_rates_item1[2][i] +
                            daily_customers[3] * conversion_rates_item1[3][i]) * prices[i] + self.item2.get_price() * (
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item2[0][0] * weights[0][0] +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item2[0][1] * weights[0][1] +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item2[0][2] * weights[0][2] +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item2[0][3] * weights[0][3] +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item2[1][0] * weights[1][0] * (1-self.discounts[1]) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item2[1][1] * weights[1][1] * (1-self.discounts[1]) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item2[1][2] * weights[1][2] * (1-self.discounts[1]) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item2[1][3] * weights[1][3] * (1-self.discounts[1]) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item2[2][0] * weights[2][0] * (1-self.discounts[2]) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item2[2][1] * weights[2][1] * (1-self.discounts[2]) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item2[2][2] * weights[2][2] * (1-self.discounts[2]) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item2[2][3] * weights[2][3] * (1-self.discounts[2]) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item2[3][0] * weights[3][0] * (1-self.discounts[3]) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item2[3][1] * weights[3][1] * (1-self.discounts[3]) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item2[3][2] * weights[3][2] * (1-self.discounts[3]) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item2[3][3] * weights[3][3] * (1-self.discounts[3]))

        # Optimal objective value, useful to compute the regret later
        opt = max(objective)

        # Reward obtained when buying item 2 (one element per arm)
        reward_item2 = np.zeros(4)
        for i in range(4):
            reward_item2[i] = self.item2.get_price() * (
                    conversion_rates_item2[0][i] * weights[0][i] +
                    conversion_rates_item2[1][i] * weights[1][i] * (1-self.discounts[1]) +
                    conversion_rates_item2[2][i] * weights[2][i] * (1-self.discounts[2]) +
                    conversion_rates_item2[3][i] * weights[3][i] * (1-self.discounts[3]))

        # Launching the experiments, using both UCB1 and Thompson Sampling to learn the price of item 1
        n_experiments = 100
        T = 1000
        ucb1_rewards_per_experiment = []
        ts_rewards_per_experiment = []

        for e in range(n_experiments):
            print(e+1)
            env = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)
            ucb1_learner = UCB1_item1(n_arms, daily_customers, prices, reward_item2)
            ts_learner = TS_Learner_item1(n_arms, daily_customers, prices, reward_item2)

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
        # Number of arms (computed as np.ceil((T * np.log(T))**(1/4)).astype(int))
        n_arms = 9

        # Candidate prices (one per arm)
        prices = np.array([50, 100, 150, 200, 300, 400, 450, 500, 550])

        # Conversion rates for item 1 (one per arm)
        conversion_rates_item1 = np.array([[0.9, 0.84, 0.72, 0.59, 0.50, 0.42, 0.23, 0.13, 0.07],
                                           [0.87, 0.75, 0.57, 0.44, 0.36, 0.29, 0.13, 0.10, 0.02],
                                           [0.89, 0.78, 0.62, 0.48, 0.45, 0.36, 0.17, 0.12, 0.05],
                                           [0.88, 0.78, 0.59, 0.44, 0.37, 0.31, 0.15, 0.13, 0.03]])

        # Conversion rates for item 2 given item 1 and promo (row: promo; column: customer class)
        conversion_rates_item2 = self.data.get_conversion_rates_item21()

        # Number of daily customers (one per class)
        daily_customers = self.data.get_daily_customers()

        # Promo assigment (row: promo; column: customer class) - Taken by step 1
        # TODO use also the matrix of the second experiment of Step 1 (make two experiments)
        weights = np.array([[0.92553191, 1, 0, 1],
                            [0, 0, 0.74339623, 0],
                            [0, 0, 0.25660377, 0],
                            [0.07446809, 0, 0, 0]])

        # Objective array (one element per arm)
        objective = np.zeros(n_arms)
        for i in range(n_arms):
            objective[i] = (daily_customers[0] * conversion_rates_item1[0][i] +
                            daily_customers[1] * conversion_rates_item1[1][i] +
                            daily_customers[2] * conversion_rates_item1[2][i] +
                            daily_customers[3] * conversion_rates_item1[3][i]) * prices[i] + self.item2.get_price() * (
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item2[0][0] * weights[0][0] +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item2[0][1] * weights[0][1] +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item2[0][2] * weights[0][2] +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item2[0][3] * weights[0][3] +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item2[1][0] * weights[1][0] * (1-self.discounts[1]) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item2[1][1] * weights[1][1] * (1-self.discounts[1]) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item2[1][2] * weights[1][2] * (1-self.discounts[1]) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item2[1][3] * weights[1][3] * (1-self.discounts[1]) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item2[2][0] * weights[2][0] * (1-self.discounts[2]) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item2[2][1] * weights[2][1] * (1-self.discounts[2]) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item2[2][2] * weights[2][2] * (1-self.discounts[2]) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item2[2][3] * weights[2][3] * (1-self.discounts[2]) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item2[3][0] * weights[3][0] * (1-self.discounts[3]) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item2[3][1] * weights[3][1] * (1-self.discounts[3]) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item2[3][2] * weights[3][2] * (1-self.discounts[3]) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item2[3][3] * weights[3][3] * (1-self.discounts[3]))

        # Optimal objective value, useful to compute the regret later
        opt = max(objective)

        # Launching the experiments, using UCB1 to learn the conversion rates of item 2
        # and both UCB1 and Thompson Sampling to learn the price of item 1
        n_experiments = 100
        T = 1000
        ucb1_rewards_per_experiment_item1 = []
        ts_rewards_per_experiment_item1 = []

        for e in range(n_experiments):
            print(e + 1)

            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers()
            daily_customers_empirical_means = np.zeros(4)

            env_item2_class1 = Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 0])
            env_item2_class2 = Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 1])
            env_item2_class3 = Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 2])
            env_item2_class4 = Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 3])
            ucb1_learner_item2_class1 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class2 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class3 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class4 = UCB1_item2(n_arms=4)

            env_item1 = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)
            ucb1_learner_item1 = UCB1_item1(n_arms, daily_customers, prices, reward_item2=np.zeros(4))
            ts_learner_item1 = TS_Learner_item1(n_arms, daily_customers, prices, reward_item2=np.zeros(4))

            for t in range(0, T):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(
                    empirical_means=daily_customers_empirical_means, sample=daily_customers_sample)

                # Learning the conversion rates for item 2 / customer class 1
                pulled_arm = ucb1_learner_item2_class1.pull_arm()
                reward = env_item2_class1.round(pulled_arm)
                ucb1_learner_item2_class1.update(pulled_arm, reward)

                # Learning the conversion rates for item 2 / customer class 2
                pulled_arm = ucb1_learner_item2_class2.pull_arm()
                reward = env_item2_class2.round(pulled_arm)
                ucb1_learner_item2_class2.update(pulled_arm, reward)

                # Learning the conversion rates for item 2 / customer class 3
                pulled_arm = ucb1_learner_item2_class3.pull_arm()
                reward = env_item2_class3.round(pulled_arm)
                ucb1_learner_item2_class3.update(pulled_arm, reward)

                # Learning the conversion rates for item 2 / customer class 4
                pulled_arm = ucb1_learner_item2_class4.pull_arm()
                reward = env_item2_class4.round(pulled_arm)
                ucb1_learner_item2_class4.update(pulled_arm, reward)

                # Retrieving the conversion rates in the usual 4x4 matrix (row: promo; column: customer class)
                conversion_rates_item2_em = np.zeros([4, 4])
                conversion_rates_item2_em[:, 0] = ucb1_learner_item2_class1.get_empirical_means()
                conversion_rates_item2_em[:, 1] = ucb1_learner_item2_class2.get_empirical_means()
                conversion_rates_item2_em[:, 2] = ucb1_learner_item2_class3.get_empirical_means()
                conversion_rates_item2_em[:, 3] = ucb1_learner_item2_class4.get_empirical_means()

                # Computing the reward obtained when buying item 2 (one element per arm)
                reward_item2 = np.zeros(4)
                for i in range(4):
                    reward_item2[i] = self.item2.get_price() * (
                            conversion_rates_item2_em[0][i] * weights[0][i] +
                            conversion_rates_item2_em[1][i] * weights[1][i] * (1-self.discounts[1]) +
                            conversion_rates_item2_em[2][i] * weights[2][i] * (1-self.discounts[2]) +
                            conversion_rates_item2_em[3][i] * weights[3][i] * (1-self.discounts[3]))

                # Updating the daily customers and the reward of item 2 in the learners for the price of item 1
                ucb1_learner_item1.update_daily_customers(daily_customers_empirical_means)
                ts_learner_item1.update_daily_customers(daily_customers_empirical_means)
                ucb1_learner_item1.update_reward_item2(reward_item2)
                ts_learner_item1.update_reward_item2(reward_item2)

                # Learning the price of item 1 (UCB1)
                pulled_arm = ucb1_learner_item1.pull_arm()
                reward = env_item1.round(pulled_arm)
                ucb1_learner_item1.update(pulled_arm, reward)

                # Learning the price of item 1 (Thompson Sampling)
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
    # TODO add the daily customer learning also here (UCB_matching and final rew_UCB "normalization" need to be fixed)
    #   Add daily_customers to UCB_matching (both in input parameters and in upper_bound computation
    #   Multiply for the number of daily customers in the final rew_UCB.append(...)
    #   Of course, add learner and environment for daily_customers, as well as sampling and mean
    def simulation_step_5(self, p1_frac, p2_frac, p3_frac):
        # Conversion rates for item 2 given item 1 and promo (row: promo; column: customer class)
        conversion_rates_item21 = self.data.get_conversion_rates_item21()

        # Number of daily customers (one per class)
        daily_customers = np.ones(4)

        # TODO create this array in the main and pass it to this function
        p_frac = np.array([p1_frac, p2_frac, p3_frac])

        # Objective matrix (row: promo; column: customer class)
        # Assumption: the number of promos of every type of promo is smaller than the number of customers of every class
        objective = np.zeros((3, 4))
        for i in range(3):
            objective[i][:] = self.item2.get_price() * (1-self.discounts[i+1]) * conversion_rates_item21[i+1][:] * p_frac[i]

        # Optimal objective value, useful to compute the regret later
        opt = linear_sum_assignment(-objective)

        # Launching the experiments, using UCB1 to learn the conversion rates of item 2 and the matching
        n_exp = 50
        T = 5000
        regret_ucb = np.zeros((n_exp, T))
        reward_ucb = []
        for e in range(n_exp):
            print(e + 1)

            #env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            #learner_daily_customers = Learner_Customers()
            #daily_customers_empirical_means = np.zeros(4)

            env_item2_class1 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 0])
            env_item2_class2 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 1])
            env_item2_class3 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 2])
            env_item2_class4 = Environment_First(n_arms=4, probabilities=conversion_rates_item21[:, 3])
            ucb1_learner_item2_class1 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class2 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class3 = UCB1_item2(n_arms=4)
            ucb1_learner_item2_class4 = UCB1_item2(n_arms=4)

            probabilities = np.zeros((3, 4))
            env = Environment_First(probabilities.size, probabilities)
            learner = UCB_Matching(probabilities.size, *probabilities.shape, self.item2.get_price(), daily_customers, 1-self.discounts, p_frac)

            rew_UCB = []
            opt_rew = []

            for t in range(T):
                # Learning the number of customers
                #daily_customers_sample = env_daily_customers.sample()
                #daily_customers_empirical_means = learner_daily_customers.update_daily_customers(
                #    empirical_means=daily_customers_empirical_means, sample=daily_customers_sample)

                # Learning the conversion rates for item 2 / customer class 1
                pulled_arm = ucb1_learner_item2_class1.pull_arm()
                reward = env_item2_class1.round(pulled_arm)
                ucb1_learner_item2_class1.update(pulled_arm, reward)

                # Learning the conversion rates for item 2 / customer class 2
                pulled_arm = ucb1_learner_item2_class2.pull_arm()
                reward = env_item2_class2.round(pulled_arm)
                ucb1_learner_item2_class2.update(pulled_arm, reward)

                # Learning the conversion rates for item 2 / customer class 3
                pulled_arm = ucb1_learner_item2_class3.pull_arm()
                reward = env_item2_class3.round(pulled_arm)
                ucb1_learner_item2_class3.update(pulled_arm, reward)

                # Learning the conversion rates for item 2 / customer class 4
                pulled_arm = ucb1_learner_item2_class4.pull_arm()
                reward = env_item2_class4.round(pulled_arm)
                ucb1_learner_item2_class4.update(pulled_arm, reward)

                # Retrieving the conversion rates in a 3x4 matrix (row: promo; column: customer class)
                # As always, we do not consider promo P0 for the matching, since it corresponds to "no discount"
                probabilities[:, 0] = ucb1_learner_item2_class1.get_empirical_means()[1:]
                probabilities[:, 1] = ucb1_learner_item2_class2.get_empirical_means()[1:]
                probabilities[:, 2] = ucb1_learner_item2_class3.get_empirical_means()[1:]
                probabilities[:, 3] = ucb1_learner_item2_class4.get_empirical_means()[1:]

                # Updating the probability array in the environment
                env.set_probabilities(probabilities)

                # Learning the matching using UCB1
                pulled_arms = learner.pull_arm()
                rewards = env.round(pulled_arms)
                learner.update(pulled_arms, rewards)

                rew_UCB.append((rewards * self.item2.get_price() * (1-self.discounts[1:]) * p_frac).sum())
                opt_rew.append(objective[opt].sum())

            regret_ucb[e, :] = np.cumsum(opt_rew) - np.cumsum(rew_UCB)
            reward_ucb.append(rew_UCB)

        # Plotting the regret and the reward
        plt.figure(0)
        plt.xlabel('t')
        plt.ylabel('Regret')
        plt.plot(regret_ucb.mean(axis=0), "b")
        plt.title('STEP 5')
        plt.show()

        plt.figure(1)
        plt.xlabel('t')
        plt.ylabel('Reward')
        plt.plot(np.mean(reward_ucb, axis=0), "b")
        plt.title('STEP 5')
        plt.show()

########################################################################################################################

    def simulation_step_6(self, p0_frac, p1_frac, p2_frac, p3_frac):
        # Candidate prices (one per arm) - The central ones (€300 and €50) is taken by step 1
        prices_item1 = np.array([50, 100, 150, 200, 300, 400, 450, 500, 550])
        prices_item2 = np.array([40, 50, 60])

        # Conversion rates for item 1 (one per arm)
        conversion_rates_item1 = np.array([[0.9, 0.84, 0.72, 0.59, 0.50, 0.42, 0.23, 0.13, 0.07],
                                           [0.87, 0.75, 0.57, 0.44, 0.36, 0.29, 0.13, 0.10, 0.02],
                                           [0.89, 0.78, 0.62, 0.48, 0.45, 0.36, 0.17, 0.12, 0.05],
                                           [0.88, 0.78, 0.59, 0.44, 0.37, 0.31, 0.15, 0.13, 0.03]])

        # Conversion rates for item 2 (taken from the form)
        conversion_rates_item21 = self.data.get_conversion_rates_item21()

        # Conversion rates for item 2 (one per arm)
        conversion_rates_item21_by_price = np.array([[self.data.get_conversion_rates_item21() + 0.1],
                                                     [self.data.get_conversion_rates_item21()],
                                                     [self.data.get_conversion_rates_item21() - 0.1]])

        # Number of daily customers per class # TODO keep or remove as in previous steps?
        daily_customers = np.ones(4)

        # Array of 1-discount percentages
        discounts = np.array([1-self.discounts[1], 1-self.discounts[2], 1-self.discounts[3]])
        promos = np.array([1, 1-self.discounts[1], 1-self.discounts[2], 1-self.discounts[3]])

        # Array of promo fractions
        p_frac = np.array([p1_frac, p2_frac, p3_frac])

        # Number of arms for pricing item 1
        n_arms = 9

        # Matching weights taken by step 1
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
                            daily_customers[3] * conversion_rates_item1[3][i]) * prices_item1[i] + self.item2.get_price() * (
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[0][0] * weights[0][0] +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[0][1] * weights[0][1] +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[0][2] * weights[0][2] +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[0][3] * weights[0][3] +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[1][0] * weights[1][0] * (1-self.discounts[1]) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[1][1] * weights[1][1] * (1-self.discounts[1]) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[1][2] * weights[1][2] * (1-self.discounts[1]) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[1][3] * weights[1][3] * (1-self.discounts[1]) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[2][0] * weights[2][0] * (1-self.discounts[2]) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[2][1] * weights[2][1] * (1-self.discounts[2]) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[2][2] * weights[2][2] * (1-self.discounts[2]) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[2][3] * weights[2][3] * (1-self.discounts[2]) +
                            daily_customers[0] * conversion_rates_item1[0][i] * conversion_rates_item21[3][0] * weights[3][0] * (1-self.discounts[3]) +
                            daily_customers[1] * conversion_rates_item1[1][i] * conversion_rates_item21[3][1] * weights[3][1] * (1-self.discounts[3]) +
                            daily_customers[2] * conversion_rates_item1[2][i] * conversion_rates_item21[3][2] * weights[3][2] * (1-self.discounts[3]) +
                            daily_customers[3] * conversion_rates_item1[3][i] * conversion_rates_item21[3][3] * weights[3][3] * (1-self.discounts[3]))

        # Storing the optimal objective value to compute the regret later
        opt = max(objective)

        n_exp = 20
        T = 5000

        ucb1_rewards_per_experiment_item1 = []
        ts_rewards_per_experiment_item1 = []

        for e in range(n_exp):
            print(e + 1)

            env_item2_class1 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, :, 0].flatten())
            env_item2_class2 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, :, 1].flatten())
            env_item2_class3 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, :, 2].flatten())
            env_item2_class4 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, :, 3].flatten())

            ucb1_learner_item2_class1 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class2 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class3 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class4 = UCB1_item2(n_arms=12)

            probabilities = np.zeros((3, 4))
            ucb1_learner_matching = UCB_Matching(probabilities.size, *probabilities.shape, price=0, daily_customers=daily_customers, discounts=discounts, p_frac=p_frac)
            env_matching = Environment_First(probabilities.size, probabilities)

            env_item1 = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)

            ucb1_learner_item1 = UCB1_item1(n_arms=n_arms, daily_customers=daily_customers, prices=prices_item1, reward_item2=np.zeros(4))
            ts_learner_item1 = TS_Learner_item1(n_arms=n_arms, daily_customers=daily_customers, prices=prices_item1, reward_item2=np.zeros(4))


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
                argmax_index_item2_class1 = np.argmax([sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                      sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * promos),
                                                      sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[2] * prices_item2[2]* promos)])
                argmax_index_item2_class2 = np.argmax([sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                      sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * promos),
                                                      sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[2] * prices_item2[2] * promos)])
                argmax_index_item2_class3 = np.argmax([sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                      sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * promos),
                                                      sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[2] * prices_item2[2] * promos)])
                argmax_index_item2_class4 = np.argmax([sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                      sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * promos),
                                                      sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[2] * prices_item2[2] * promos)])

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
                for i in range(4):
                    reward_item2[i] = self.item2.get_price() * (
                            conversion_rates_item2_ub[0][i] * weights[0][i] +
                            conversion_rates_item2_ub[1][i] * weights[1][i] * (1 - self.discounts[1]) +
                            conversion_rates_item2_ub[2][i] * weights[2][i] * (1 - self.discounts[2]) +
                            conversion_rates_item2_ub[3][i] * weights[3][i] * (1 - self.discounts[3]))

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

########################################################################################################################

    def simulation_step_7(self, p0_frac, p1_frac, p2_frac, p3_frac):
        # Candidate prices (one per arm) - The central ones (€300 and €50) is taken by step 1
        prices_item1 = np.array([50, 100, 150, 200, 300, 400, 450, 500, 550])
        prices_item2 = np.array([40, 50, 60])
        n_phases = 3
        T = 5000
        phases_len = int(T / n_phases)
        window_size = int(np.sqrt(T))
        n_exp = 10

        # Conversion rates for item 1 (one per arm)
        conversion_rates_item1 = np.array([[0.9, 0.84, 0.72, 0.59, 0.50, 0.42, 0.23, 0.13, 0.07],
                                           [0.87, 0.75, 0.57, 0.44, 0.36, 0.29, 0.13, 0.10, 0.02],
                                           [0.89, 0.78, 0.62, 0.48, 0.45, 0.36, 0.17, 0.12, 0.05],
                                           [0.88, 0.78, 0.59, 0.44, 0.37, 0.31, 0.15, 0.13, 0.03]])

        conversion_rates_item1_NS = np.array([conversion_rates_item1 + 0.2,
                                              conversion_rates_item1,
                                              conversion_rates_item1 - 0.2])
        conversion_rates_item1_NS = np.clip(conversion_rates_item1_NS, 0, 1)
        # dim (5,1,4,9): 5 phases - 4 customers classes - 9 prices

        # Conversion rates for item 2 (taken from the form)
        conversion_rates_item21 = self.data.get_conversion_rates_item21()

        # Conversion rates for item 2 (one per arm)
        conversion_rates_item21_by_price = np.array([self.data.get_conversion_rates_item21() + 0.05,
                                                     self.data.get_conversion_rates_item21(),
                                                     self.data.get_conversion_rates_item21() - 0.05])

        conversion_rates_item21_by_price_NS = np.array([conversion_rates_item21_by_price + 0.2,
                                                        conversion_rates_item21_by_price,
                                                        conversion_rates_item21_by_price - 0.2])
        conversion_rates_item21_by_price_NS = np.clip(conversion_rates_item21_by_price_NS, 0, 1)
        # dim (5,1,3,1,4,4): 5 phases - 3 full prices (40€-50€-60€) - 4 promos - 4 customers classes

        # Number of daily customers per class # TODO keep or remove as in previous steps?
        daily_customers = np.ones(4)

        # Array of 1-discount percentages
        discounts = np.array([1-self.discounts[1], 1-self.discounts[2], 1-self.discounts[3]])
        promos = np.array([1, 1-self.discounts[1], 1-self.discounts[2], 1-self.discounts[3]])

        # Array of promo fractions
        p_frac = np.array([p1_frac, p2_frac, p3_frac])

        # Number of arms for pricing item 1
        n_arms = 9

        # Matching weights taken by step 1
        weights = np.array([[0.92553191, 1, 0, 1],
                            [0, 0, 0.74339623, 0],
                            [0, 0, 0.25660377, 0],
                            [0.07446809, 0, 0, 0]])

        # Computing the objective array (one element per arm)
        objective = np.zeros([n_arms, n_phases, 3])
        for i in range(n_arms):
            for j in range(n_phases):
                for k in range(3):
                    objective[i, j, k] = (daily_customers[0] * conversion_rates_item1_NS[j, 0, i] +
                                    daily_customers[1] * conversion_rates_item1_NS[j, 1, i] +
                                    daily_customers[2] * conversion_rates_item1_NS[j, 2, i] +
                                    daily_customers[3] * conversion_rates_item1_NS[j, 3, i]) * prices_item1[i] + prices_item2[k] * (
                                    daily_customers[0] * conversion_rates_item1_NS[j, 0, i] * conversion_rates_item21_by_price_NS[j, k, 0, 0] * weights[0][0] +
                                    daily_customers[1] * conversion_rates_item1_NS[j, 1, i] * conversion_rates_item21_by_price_NS[j, k, 0, 1] * weights[0][1] +
                                    daily_customers[2] * conversion_rates_item1_NS[j, 2, i] * conversion_rates_item21_by_price_NS[j, k, 0, 2] * weights[0][2] +
                                    daily_customers[3] * conversion_rates_item1_NS[j, 3, i] * conversion_rates_item21_by_price_NS[j, k, 0, 3] * weights[0][3] +
                                    daily_customers[0] * conversion_rates_item1_NS[j, 0, i] * conversion_rates_item21_by_price_NS[j, k, 1, 0] * weights[1][0] * (1-self.discounts[0]) +
                                    daily_customers[1] * conversion_rates_item1_NS[j, 1, i] * conversion_rates_item21_by_price_NS[j, k, 1, 1] * weights[1][1] * (1-self.discounts[0]) +
                                    daily_customers[2] * conversion_rates_item1_NS[j, 2, i] * conversion_rates_item21_by_price_NS[j, k, 1, 2] * weights[1][2] * (1-self.discounts[0]) +
                                    daily_customers[3] * conversion_rates_item1_NS[j, 3, i] * conversion_rates_item21_by_price_NS[j, k, 1, 3] * weights[1][3] * (1-self.discounts[0]) +
                                    daily_customers[0] * conversion_rates_item1_NS[j, 0, i] * conversion_rates_item21_by_price_NS[j, k, 2, 0] * weights[2][0] * (1-self.discounts[1]) +
                                    daily_customers[1] * conversion_rates_item1_NS[j, 1, i] * conversion_rates_item21_by_price_NS[j, k, 2, 1] * weights[2][1] * (1-self.discounts[1]) +
                                    daily_customers[2] * conversion_rates_item1_NS[j, 2, i] * conversion_rates_item21_by_price_NS[j, k, 2, 2] * weights[2][2] * (1-self.discounts[1]) +
                                    daily_customers[3] * conversion_rates_item1_NS[j, 3, i] * conversion_rates_item21_by_price_NS[j, k, 2, 3] * weights[2][3] * (1-self.discounts[1]) +
                                    daily_customers[0] * conversion_rates_item1_NS[j, 0, i] * conversion_rates_item21_by_price_NS[j, k, 3, 0] * weights[3][0] * (1-self.discounts[2]) +
                                    daily_customers[1] * conversion_rates_item1_NS[j, 1, i] * conversion_rates_item21_by_price_NS[j, k, 3, 1] * weights[3][1] * (1-self.discounts[2]) +
                                    daily_customers[2] * conversion_rates_item1_NS[j, 2, i] * conversion_rates_item21_by_price_NS[j, k, 3, 2] * weights[3][2] * (1-self.discounts[2]) +
                                    daily_customers[3] * conversion_rates_item1_NS[j, 3, i] * conversion_rates_item21_by_price_NS[j, k, 3, 3] * weights[3][3] * (1-self.discounts[2]))

        # Storing the optimal objective value to compute the regret later
        opt = np.amax(np.amax(objective, axis=2), axis=0)

        ucb1_rewards_per_experiment_item1 = []
        ts_rewards_per_experiment_item1 = []
        ts_rewards_per_experiment_item1_NS = []


        for e in range(n_exp):
            print(e + 1)

            env_item2_class1 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, 0].flatten())
            env_item2_class2 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, 1].flatten())
            env_item2_class3 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, 2].flatten())
            env_item2_class4 = Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price[:, :, 3].flatten())

            env_item2_class1_NS = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price_NS[:, :, :, 0].reshape(n_phases, 12), horizon=T)
            env_item2_class2_NS = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price_NS[:, :, :, 1].reshape(n_phases, 12), horizon=T)
            env_item2_class3_NS = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price_NS[:, :, :, 2].reshape(n_phases, 12), horizon=T)
            env_item2_class4_NS = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item21_by_price_NS[:, :, :, 3].reshape(n_phases, 12), horizon=T)

            ucb1_learner_item2_class1 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class2 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class3 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class4 = UCB1_item2(n_arms=12)

            SWTS_learner_item2_class1 = SWTS_Learner_item2(n_arms=12, window_size=window_size)
            SWTS_learner_item2_class2 = SWTS_Learner_item2(n_arms=12, window_size=window_size)
            SWTS_learner_item2_class3 = SWTS_Learner_item2(n_arms=12, window_size=window_size)
            SWTS_learner_item2_class4 = SWTS_Learner_item2(n_arms=12, window_size=window_size)

            probabilities = np.zeros((3, 4))
            #probabilities_NS = np.zeros((5, 3, 4))

            ucb1_learner_matching = UCB_Matching(probabilities.size, *probabilities.shape, price=0, daily_customers=daily_customers, discounts=discounts, p_frac=p_frac)
            env_matching = Environment_First(probabilities.size, probabilities)

            env_item1 = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)
            env_item1_NS = Non_Stationary_Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1_NS, horizon=T)

            ucb1_learner_item1 = UCB1_item1(n_arms=n_arms, daily_customers=daily_customers, prices=prices_item1, reward_item2=np.zeros(4))
            ts_learner_item1 = TS_Learner_item1(n_arms=n_arms, daily_customers=daily_customers, prices=prices_item1, reward_item2=np.zeros(4))

            ts_learner_item1_NS = SWTS_Learner(n_arms=n_arms, daily_customers=daily_customers, prices=prices_item1, reward_item2=np.zeros(4), window_size=window_size)


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

                # Non stationary
                # Item 2 Class 1
                pulled_arm = SWTS_learner_item2_class1.pull_arm()
                reward = env_item2_class1_NS.round(pulled_arm)
                SWTS_learner_item2_class1.update(pulled_arm, reward)

                # Item 2 CLass 2
                pulled_arm = SWTS_learner_item2_class2.pull_arm()
                reward = env_item2_class2_NS.round(pulled_arm)
                SWTS_learner_item2_class2.update(pulled_arm, reward)

                # Item 2 Class 3
                pulled_arm = SWTS_learner_item2_class3.pull_arm()
                reward = env_item2_class3_NS.round(pulled_arm)
                SWTS_learner_item2_class3.update(pulled_arm, reward)

                # Item 2 Class 4
                pulled_arm = SWTS_learner_item2_class4.pull_arm()
                reward = env_item2_class4_NS.round(pulled_arm)
                SWTS_learner_item2_class4.update(pulled_arm, reward)

                # Matching
                conversion_rates_item2_ub = np.zeros([4, 4])
                argmax_index_item2_class1 = np.argmax([sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                      sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * promos),
                                                      sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[2] * prices_item2[2]* promos)])
                argmax_index_item2_class2 = np.argmax([sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                      sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * promos),
                                                      sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[2] * prices_item2[2] * promos)])
                argmax_index_item2_class3 = np.argmax([sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                      sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * promos),
                                                      sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[2] * prices_item2[2] * promos)])
                argmax_index_item2_class4 = np.argmax([sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                      sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * promos),
                                                      sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[2] * prices_item2[2] * promos)])

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
                for i in range(4):
                    reward_item2[i] = self.item2.get_price() * (
                            conversion_rates_item2_ub[0][i] * weights[0][i] +
                            conversion_rates_item2_ub[1][i] * weights[1][i] * (1 - self.discounts[1]) +
                            conversion_rates_item2_ub[2][i] * weights[2][i] * (1 - self.discounts[2]) +
                            conversion_rates_item2_ub[3][i] * weights[3][i] * (1 - self.discounts[3]))

                ucb1_learner_item1.update_reward_item2(reward_item2)
                ts_learner_item1.update_reward_item2(reward_item2)

                pulled_arm = ucb1_learner_item1.pull_arm()
                reward = env_item1.round(pulled_arm)
                ucb1_learner_item1.update(pulled_arm, reward)

                pulled_arm = ts_learner_item1.pull_arm()
                reward = env_item1.round(pulled_arm)
                ts_learner_item1.update(pulled_arm, reward)


                # Non stationary
                # Matching
                conversion_rates_item2_ub = np.zeros([4, 4])
                argmax_index_item2_class1 = np.argmax([sum(SWTS_learner_item2_class1.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                       sum(SWTS_learner_item2_class1.get_empirical_means().reshape(3, 4)[0] * prices_item2[1] * promos),
                                                       sum(SWTS_learner_item2_class1.get_empirical_means().reshape(3, 4)[0] * prices_item2[2] * promos)])
                argmax_index_item2_class2 = np.argmax([sum(SWTS_learner_item2_class2.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                       sum(SWTS_learner_item2_class2.get_empirical_means().reshape(3, 4)[0] * prices_item2[1] * promos),
                                                       sum(SWTS_learner_item2_class2.get_empirical_means().reshape(3, 4)[0] * prices_item2[2] * promos)])
                argmax_index_item2_class3 = np.argmax([sum(SWTS_learner_item2_class3.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                       sum(SWTS_learner_item2_class3.get_empirical_means().reshape(3, 4)[0] * prices_item2[1] * promos),
                                                       sum(SWTS_learner_item2_class3.get_empirical_means().reshape(3, 4)[0] * prices_item2[2] * promos)])
                argmax_index_item2_class4 = np.argmax([sum(SWTS_learner_item2_class4.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * promos),
                                                       sum(SWTS_learner_item2_class4.get_empirical_means().reshape(3, 4)[0] * prices_item2[1] * promos),
                                                       sum(SWTS_learner_item2_class4.get_empirical_means().reshape(3, 4)[0] * prices_item2[2] * promos)])

                majority_voting = stats.mode([argmax_index_item2_class1, argmax_index_item2_class2, argmax_index_item2_class3,argmax_index_item2_class4])[0][0]

                conversion_rates_item2_ub[:, 0] = SWTS_learner_item2_class1.get_empirical_means().reshape(3, 4)[majority_voting]
                conversion_rates_item2_ub[:, 1] = SWTS_learner_item2_class2.get_empirical_means().reshape(3, 4)[majority_voting]
                conversion_rates_item2_ub[:, 2] = SWTS_learner_item2_class3.get_empirical_means().reshape(3, 4)[majority_voting]
                conversion_rates_item2_ub[:, 3] = SWTS_learner_item2_class4.get_empirical_means().reshape(3, 4)[majority_voting]

                probabilities_NS = np.zeros((3, 4))
                for i in range(3):
                    for j in range(4):
                        probabilities_NS[i][j] = conversion_rates_item2_ub[i + 1, j]

                env_matching.set_probabilities(probabilities_NS)
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
                for i in range(4):
                    reward_item2[i] = self.item2.get_price() * (
                            conversion_rates_item2_ub[0][i] * weights[0][i] +
                            conversion_rates_item2_ub[1][i] * weights[1][i] * (1 - self.discounts[1]) +
                            conversion_rates_item2_ub[2][i] * weights[2][i] * (1 - self.discounts[2]) +
                            conversion_rates_item2_ub[3][i] * weights[3][i] * (1 - self.discounts[3]))

                ts_learner_item1_NS.update_reward_item2(reward_item2)

                pulled_arm = ts_learner_item1_NS.pull_arm()
                reward = env_item1_NS.round(pulled_arm)
                ts_learner_item1_NS.update(pulled_arm, reward)

            ucb1_rewards_per_experiment_item1.append(ucb1_learner_item1.collected_rewards)
            ts_rewards_per_experiment_item1.append(ts_learner_item1.collected_rewards)
            ts_rewards_per_experiment_item1_NS.append(ts_learner_item1_NS.collected_rewards)

        ts_instantaneous_regret = np.zeros(T)
        swts_instantaneous_regret = np.zeros(T)
        optimum_per_round = np.zeros(T)

        for i in range(0, n_phases):
            t_index = range(i * phases_len, (i + 1) * phases_len)
            optimum_per_round[t_index] = opt[i]
            ts_instantaneous_regret[t_index] = opt[i] - np.mean(ts_rewards_per_experiment_item1, axis=0)[t_index]
            swts_instantaneous_regret[t_index] = opt[i] - np.mean(ts_rewards_per_experiment_item1_NS, axis=0)[t_index]


        # Plotting the regret and the reward
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Reward")
        plt.plot(np.mean(ts_rewards_per_experiment_item1, axis=0), 'r')
        plt.plot(np.mean(ts_rewards_per_experiment_item1_NS, axis=0), 'b')
        plt.plot(optimum_per_round, '--k')
        plt.legend(["TS", "SW-TS", "Optimum"], title="STEP 7")
        plt.show()

        plt.figure(1)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
        plt.plot(np.cumsum(swts_instantaneous_regret), 'b')
        plt.legend(["TS", "SW-TS"], title="STEP 7")
        plt.show()
