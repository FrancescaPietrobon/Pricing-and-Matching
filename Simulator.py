import matplotlib.pyplot as plt
from numpy.random import normal
from scipy import stats

from Data import *
from Environment import *
from Learners.Learner_Customers import *
from Learners.Learner_Conversion_Rates_Item1 import *
from Learners.Learner_Conversion_Rates_Item2 import *
from Learners.Learner_Matching import *
from Learners.UCB1_Item1 import *
from Learners.UCB1_Items_Matching import *
from Learners.CD_UCB1_Items_Matching import CD_UCB1_Items_Matching
from Learners.TS_Item1 import *
from SWTS_Learner import *
from SWTS_Learner_item2 import *


np.random.seed(1234)


class Simulator:
    def __init__(self):
        self.discounts = np.array([0, 0.1, 0.2, 0.5])
        self.data = Data()

########################################################################################################################

    def simulation_step_1(self, promo_fractions):
        margins_item1 = self.data.margins_item1
        margins_item2 = self.data.margins_item2
        conversion_rates_item1 = self.data.conversion_rates_item1
        conversion_rates_item2 = self.data.conversion_rates_item2
        daily_customers = self.data.daily_customers

        objective = np.zeros((len(margins_item1), len(margins_item2)))
        for margin1 in range(len(margins_item1)):
            for margin2 in range(len(margins_item2)):
                daily_promos = (promo_fractions * sum(daily_customers * conversion_rates_item1[:, margin1])).astype(int)
                objective[margin1][margin2] = margins_item1[margin1] * (daily_customers * conversion_rates_item1[:, margin1]).sum() +\
                                              lp.matching_lp(margins_item2[margin2], self.discounts,
                                                             conversion_rates_item2[margin2], daily_promos,
                                                             (daily_customers * conversion_rates_item1[:, margin1]).astype(int))[0]

        idx_best_margin_item1, idx_best_margin_item2 = np.unravel_index(np.argmax(objective), objective.shape)
        daily_promos = (promo_fractions * sum(daily_customers * conversion_rates_item1[:, idx_best_margin_item1])).astype(int)
        _, best_matching = lp.matching_lp(margins_item2[idx_best_margin_item2], self.discounts,
                                          conversion_rates_item2[idx_best_margin_item2], daily_promos,
                                          (daily_customers * conversion_rates_item1[:, idx_best_margin_item1]).astype(int))

        best_price_item1 = self.data.prices_item1[idx_best_margin_item1]
        best_price_item2 = self.data.prices_item2[idx_best_margin_item2]

        # Returning the maximum objective value obtained, along with the corresponding best price for item 1
        # and best price for item 2, and the best promo-customer class matching matrix
        return objective[idx_best_margin_item1, idx_best_margin_item2], best_price_item1, best_price_item2, best_matching

########################################################################################################################

    def simulation_step_3(self, promo_fractions):
        margins_item1 = self.data.margins_item1
        n_arms = len(margins_item1)

        opt, _, selected_price_item2, matching = self.simulation_step_1(promo_fractions)

        conversion_rates_item1 = self.data.conversion_rates_item1
        conversion_rates_item2 = self.data.conversion_rates_item2[list(self.data.prices_item2).index(selected_price_item2)]
        selected_margin_item2 = self.data.margins_item2[list(self.data.prices_item2).index(selected_price_item2)]

        daily_customers = self.data.daily_customers
        weights = normalize(matching, norm='l1', axis=0)

        # Launching the experiments, using both UCB1 and Thompson Sampling to learn the price of item 1
        n_experiments = 5
        time_horizon = 1000
        ucb1_rewards_per_experiment = []
        ts_rewards_per_experiment = []

        for e in range(n_experiments):
            print("Experiment {}/{} with {} rounds".format(e+1, n_experiments, time_horizon))

            env = Environment_Single_Price(margins_item1, selected_margin_item2, conversion_rates_item1, conversion_rates_item2, weights, daily_customers, self.discounts)
            ucb1_learner = UCB1_Item1(n_arms, margins_item1, selected_margin_item2, conversion_rates_item2, weights, daily_customers, self.discounts)
            ts_learner = TS_Item1(n_arms, margins_item1, selected_margin_item2, conversion_rates_item2, weights, daily_customers, self.discounts)

            for t in range(time_horizon):
                # UCB1 Learner
                pulled_arm = ucb1_learner.pull_arm()
                conversion_rates_item1_round, _, revenue = env.round(pulled_arm)
                ucb1_learner.update(pulled_arm, conversion_rates_item1_round, revenue)

                # Thompson Sampling Learner
                pulled_arm = ts_learner.pull_arm()
                conversion_rates_item1_round, _, revenue = env.round(pulled_arm)
                ts_learner.update(pulled_arm, conversion_rates_item1_round, revenue)

            ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
            ts_rewards_per_experiment.append(ts_learner.collected_rewards)

        return opt, ucb1_rewards_per_experiment, ts_rewards_per_experiment, time_horizon

########################################################################################################################

    def simulation_step_4(self, promo_fractions):
        margins_item1 = self.data.margins_item1
        n_arms = len(margins_item1)

        opt, _, selected_price_item2, weights = self.simulation_step_1(promo_fractions)

        conversion_rates_item1 = self.data.conversion_rates_item1
        conversion_rates_item2 = self.data.conversion_rates_item2[list(self.data.prices_item2).index(selected_price_item2)]
        selected_margin_item2 = self.data.margins_item2[list(self.data.prices_item2).index(selected_price_item2)]

        daily_customers = self.data.daily_customers
        weights = normalize(weights, norm='l1', axis=0)

        # Launching the experiments, using both UCB1 and Thompson Sampling to learn the price of item 1
        # This time, the number of customers and the conversion rates for item 2 are not known
        n_experiments = 5
        time_horizon = 1000
        ucb1_rewards_per_experiment = []
        ts_rewards_per_experiment = []

        for e in range(n_experiments):
            print("Experiment {}/{} with {} rounds".format(e+1, n_experiments, time_horizon))

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers(np.zeros(4))

            # Learner for the mean of the conversion rates of item 2
            learner_conversion_rates_item2_ucb1 = Learner_Conversion_Rates_item2()
            learner_conversion_rates_item2_ts = Learner_Conversion_Rates_item2()

            # Environment and learners (UCB1 and Thompson Sampling) for the price of item 1
            env = Environment_Single_Price(margins_item1, selected_margin_item2, conversion_rates_item1, conversion_rates_item2, weights, daily_customers, self.discounts)
            ucb1_learner = UCB1_Item1(n_arms, margins_item1, selected_margin_item2, conversion_rates_item2, weights, daily_customers, self.discounts)
            ts_learner = TS_Item1(n_arms, margins_item1, selected_margin_item2, conversion_rates_item2, weights, daily_customers, self.discounts)

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(daily_customers_sample)

                # Updating the daily customers in the learners and in the environment
                ucb1_learner.daily_customers = daily_customers_empirical_means
                ts_learner.daily_customers = daily_customers_empirical_means
                env.daily_customers = daily_customers_sample

                # UCB1 Learner
                pulled_arm = ucb1_learner.pull_arm()
                reward, conversion_rates_item2_sample, revenue = env.round(pulled_arm)
                ucb1_learner.update(pulled_arm, reward, revenue)

                conversion_rates_item2_means = learner_conversion_rates_item2_ucb1.update_conversion_rates(conversion_rates_item2_sample)
                ucb1_learner.conversion_rates_item2 = conversion_rates_item2_means

                # Thompson Sampling Learner
                pulled_arm = ts_learner.pull_arm()
                reward, conversion_rates_item2_sample, revenue = env.round(pulled_arm)
                ts_learner.update(pulled_arm, reward, revenue)

                conversion_rates_item2_means = learner_conversion_rates_item2_ts.update_conversion_rates(conversion_rates_item2_sample)
                ts_learner.conversion_rates_item2 = conversion_rates_item2_means

            ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
            ts_rewards_per_experiment.append(ts_learner.collected_rewards)

        return opt, ucb1_rewards_per_experiment, ts_rewards_per_experiment, time_horizon

########################################################################################################################

    def simulation_step_5(self, promo_fractions):
        opt, selected_price_item1, selected_price_item2, _ = self.simulation_step_1(promo_fractions)

        conversion_rates_item1 = self.data.conversion_rates_item1[:, list(self.data.prices_item1).index(selected_price_item1)]
        selected_margin_item1 = self.data.margins_item1[list(self.data.prices_item1).index(selected_price_item1)]

        conversion_rates_item2 = self.data.conversion_rates_item2[list(self.data.prices_item2).index(selected_price_item2)]
        selected_margin_item2 = self.data.margins_item2[list(self.data.prices_item2).index(selected_price_item2)]

        daily_customers = self.data.daily_customers

        # Launching the experiments, using UCB1 to learn the promo-customer class matching
        n_experiments = 1
        time_horizon = 5000
        reward_matching_per_exp = []

        for e in range(n_experiments):
            print("Experiment {}/{} with {} rounds".format(e+1, n_experiments, time_horizon))

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers(np.zeros(4))

            # Learner for the mean of the conversion rates of item 1
            learner_conversion_rates_item1 = Learner_Conversion_Rates_item1()

            # Learner for the mean of the conversion rates of item 2
            learner_conversion_rates_item2 = Learner_Conversion_Rates_item2()

            # Environment and learner for the matching between promos and customer classes
            env = Environment_Matching(selected_margin_item1, selected_margin_item2, conversion_rates_item1, conversion_rates_item2, daily_customers, self.discounts)
            matching_learner = Matching(selected_margin_item2, daily_customers, self.discounts, promo_fractions)

            rew_matching_per_round = []

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(daily_customers_sample)

                # Updating the daily customers in the learner and in the environment
                matching_learner.daily_customers = daily_customers_empirical_means
                env.daily_customers = daily_customers_sample

                # Computing the weights matrix (normalized)
                weights = matching_learner.optimize(t)

                # Getting the conversion rates from the environment and updating the learner with them
                conversion_rates_item1_sample, conversion_rates_item2_sample, revenue = env.round(weights)
                matching_learner.conversion_rates_item1 = learner_conversion_rates_item1.update_conversion_rates(conversion_rates_item1_sample)
                matching_learner.conversion_rates_item2 = learner_conversion_rates_item2.update_conversion_rates(conversion_rates_item2_sample)

                rew_matching_per_round.append(revenue)

            reward_matching_per_exp.append(rew_matching_per_round)

        return opt, reward_matching_per_exp, time_horizon

########################################################################################################################

    def simulation_step_6(self, promo_fractions):
        margins_item1 = self.data.margins_item1
        margins_item2 = self.data.margins_item2

        conversion_rates_item1 = self.data.conversion_rates_item1
        conversion_rates_item2 = self.data.conversion_rates_item2

        daily_customers = self.data.daily_customers

        opt, _, _, _ = self.simulation_step_1(promo_fractions)

        # Launching the experiments
        n_experiments = 3
        time_horizon = 1000

        reward_ucb_per_exp = []

        for e in range(n_experiments):
            print("Experiment {}/{} with {} rounds".format(e+1, n_experiments, time_horizon))

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers(np.zeros(4))

            # Environment and learner for the prices of the two items and the matching
            env = Environment_Double_Prices_Matching(margins_item1, margins_item2, conversion_rates_item1, conversion_rates_item2, daily_customers, self.discounts, promo_fractions)
            ucb_learner = UCB1_Items_Matching(margins_item1, margins_item2, daily_customers, self.discounts, promo_fractions)

            reward_ucb_per_round = []

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(daily_customers_sample)

                # Updating the number of customers
                ucb_learner.daily_customers = daily_customers_empirical_means
                env.daily_customers = daily_customers_sample

                # Learning the best prices and matching
                pulled_arm = ucb_learner.pull_arm()
                reward = env.round(pulled_arm)
                ucb_learner.update(pulled_arm[0], reward)

                reward_ucb_per_round.append(reward[2])

            reward_ucb_per_exp.append(reward_ucb_per_round)

        return opt, reward_ucb_per_exp, time_horizon

########################################################################################################################
    # TODO adapt according to step 6
    def simulation_step_7(self, promo_fractions):
        # Number of arms for pricing item 1
        n_arms = 9

        # Candidate prices for item 1 and item 2 (one per arm)
        prices_item1 = self.data.margins_item1
        prices_item2 = self.data.margins_item2

        # Conversion rates for item 1 (one per arm)
        conversion_rates_item1 = self.data.conversion_rates_item1
        '''
        conversion_rates_item1_NS = np.array([conversion_rates_item1 + 0.2,
                                              conversion_rates_item1,
                                              conversion_rates_item1 - 0.2])
        '''
        conversion_rates_item1_NS = np.array([conversion_rates_item1 + 0.2,
                                              conversion_rates_item1 - 0.2])
        conversion_rates_item1_NS = np.clip(conversion_rates_item1_NS, 0, 1)
        # dim (5,4,9): 5 phases - 4 customers classes - 9 prices

        # Conversion rates for item 2 (one 4x4 matrix per arm)
        conversion_rates_item2 = np.array([self.data.conversion_rates_item21 + 0.1,
                                           self.data.conversion_rates_item21,
                                           self.data.conversion_rates_item21 - 0.1])
        '''
        conversion_rates_item2_NS = np.array([conversion_rates_item2 + 0.2,
                                              conversion_rates_item2,
                                              conversion_rates_item2 - 0.2])
        '''
        conversion_rates_item2_NS = np.array([conversion_rates_item2 + 0.2,
                                              conversion_rates_item2 - 0.2])
        conversion_rates_item2_NS = np.clip(conversion_rates_item2_NS, 0, 1)
        # dim (5,3,4,4): 5 phases - 3 full prices (40€-50€-60€) - 4 promos - 4 customers classes

        # Number of daily customers per class
        daily_customers = self.data.daily_customers

        # Promo assigment weights (row: promo; column: customer class) - Taken by step 1
        weights = normalize(self.simulation_step_1(promo_fractions)[1], 'l1', axis=0)

        # Launching the experiments
        n_experiments = 50
        time_horizon = 1000

        # Parameters for Non-Stationary Environment
        n_phases = 2
        phases_len = int(time_horizon/n_phases)
        window_size = int(np.sqrt(time_horizon))

        daily_promos = (promo_fractions * sum(daily_customers)).astype(int)
        weights = np.zeros((n_phases, 3, 4, 4))
        for j in range(n_phases):
            for k in range(3):
                weights[j, k] = normalize(lp.matching_lp(prices_item2[k], self.discounts, conversion_rates_item2_NS[j, k], daily_promos, daily_customers)[1], 'l1', axis=0)

        # Computing the objective array (one element per arm)
        objective = np.zeros([n_arms, n_phases, 3])
        for i in range(n_arms):
            for j in range(n_phases):
                for k in range(3):
                    objective[i, j, k] = sum(prices_item1[i] * daily_customers * conversion_rates_item1_NS[j, :, i] +
                                             prices_item2[k] * daily_customers * conversion_rates_item1_NS[j, :, i] *
                                             (np.dot(1 - self.discounts, conversion_rates_item2_NS[j, k] * weights[j, k])))

        # Storing the optimal objective value to compute the regret later
        opt = np.amax(np.amax(objective, axis=2), axis=0)

        ucb1_rewards_per_experiment_item1 = []
        ts_rewards_per_experiment_item1 = []
        ts_rewards_per_experiment_item1_NS = []

        for e in range(n_experiments):
            print(e + 1)

            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers()
            daily_customers_empirical_means = np.zeros(4)

            '''
            env_item2_class1 = Environment_First(n_arms=12, probabilities=conversion_rates_item2[:, :, 0].flatten())
            env_item2_class2 = Environment_First(n_arms=12, probabilities=conversion_rates_item2[:, :, 1].flatten())
            env_item2_class3 = Environment_First(n_arms=12, probabilities=conversion_rates_item2[:, :, 2].flatten())
            env_item2_class4 = Environment_First(n_arms=12, probabilities=conversion_rates_item2[:, :, 3].flatten())
            '''

            env_item2_class1 = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 0].reshape(n_phases, 12), horizon=time_horizon)
            env_item2_class2 = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 1].reshape(n_phases, 12), horizon=time_horizon)
            env_item2_class3 = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 2].reshape(n_phases, 12), horizon=time_horizon)
            env_item2_class4 = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 3].reshape(n_phases, 12), horizon=time_horizon)

            env_item2_class1_NS = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 0].reshape(n_phases, 12), horizon=time_horizon)
            env_item2_class2_NS = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 1].reshape(n_phases, 12), horizon=time_horizon)
            env_item2_class3_NS = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 2].reshape(n_phases, 12), horizon=time_horizon)
            env_item2_class4_NS = Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 3].reshape(n_phases, 12), horizon=time_horizon)

            ucb1_learner_item2_class1 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class2 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class3 = UCB1_item2(n_arms=12)
            ucb1_learner_item2_class4 = UCB1_item2(n_arms=12)

            SWTS_learner_item2_class1 = SWTS_Learner_item2(n_arms=12, window_size=window_size)
            SWTS_learner_item2_class2 = SWTS_Learner_item2(n_arms=12, window_size=window_size)
            SWTS_learner_item2_class3 = SWTS_Learner_item2(n_arms=12, window_size=window_size)
            SWTS_learner_item2_class4 = SWTS_Learner_item2(n_arms=12, window_size=window_size)

            probabilities = np.zeros((3, 4))

            ucb1_learner_matching = UCB_Matching(probabilities.size, *probabilities.shape, price=0, daily_customers=daily_customers, discounts=1-self.discounts, p_frac=promo_fractions)
            env_matching = Environment_Matching(probabilities.size, probabilities)

            # env_item1 = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)
            env_item1 = Non_Stationary_Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1_NS, horizon=time_horizon)
            env_item1_NS = Non_Stationary_Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1_NS, horizon=time_horizon)

            ucb1_learner_item1 = UCB1_Item1(n_arms=n_arms, daily_customers=daily_customers, margins=prices_item1, reward_item2=np.zeros(4))
            ts_learner_item1 = TS_Item1(n_arms=n_arms, daily_customers=daily_customers, margins=prices_item1, reward_item2=np.zeros(4))

            ts_learner_item1_NS = SWTS_Learner(n_arms=n_arms, daily_customers=daily_customers, margins=prices_item1, reward_item2=np.zeros(4), window_size=window_size)


            for t in range(time_horizon):

                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(
                    empirical_means=daily_customers_empirical_means, sample=daily_customers_sample)

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
                argmax_index_item2_class1 = np.argmax([sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * (1-self.discounts)),
                                                      sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * (1-self.discounts)),
                                                      sum(ucb1_learner_item2_class1.get_empirical_means().reshape(3, 4)[2] * prices_item2[2]* (1-self.discounts))])
                argmax_index_item2_class2 = np.argmax([sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * (1-self.discounts)),
                                                      sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * (1-self.discounts)),
                                                      sum(ucb1_learner_item2_class2.get_empirical_means().reshape(3, 4)[2] * prices_item2[2] * (1-self.discounts))])
                argmax_index_item2_class3 = np.argmax([sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * (1-self.discounts)),
                                                      sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * (1-self.discounts)),
                                                      sum(ucb1_learner_item2_class3.get_empirical_means().reshape(3, 4)[2] * prices_item2[2] * (1-self.discounts))])
                argmax_index_item2_class4 = np.argmax([sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * (1-self.discounts)),
                                                      sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[1] * prices_item2[1] * (1-self.discounts)),
                                                      sum(ucb1_learner_item2_class4.get_empirical_means().reshape(3, 4)[2] * prices_item2[2] * (1-self.discounts))])

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
                ucb1_learner_matching.update_daily_customers(daily_customers_empirical_means)

                pulled_arms = ucb1_learner_matching.pull_arm()
                rewards = env_matching.round(pulled_arms)
                ucb1_learner_matching.update(pulled_arms, rewards)

                # Price item 1
                res = np.zeros((3, 4))

                res[pulled_arms[0][0]][pulled_arms[1][0]] = 1
                res[pulled_arms[0][1]][pulled_arms[1][1]] = 1
                res[pulled_arms[0][2]][pulled_arms[1][2]] = 1

                weights = np.zeros((4, 4))
                for i in range(0, 3):
                    for j in range(0, 4):
                        if res[i][j] == 1:
                            weights[i + 1][j] = promo_fractions[i + 1] * sum(daily_customers_empirical_means)

                for j in range(0, 4):
                    weights[0][j] = daily_customers_empirical_means[j] - sum(weights[:, j])

                weights = normalize(weights, 'l1', axis=0)

                reward_item2 = np.zeros(4)
                for i in range(4):
                    reward_item2[i] = sum(prices_item2[majority_voting] * (1 - self.discounts) * conversion_rates_item2_ub[:, i] * weights[:, i])

                '''
                ucb1_learner_item1.update_reward_item2(reward_item2)
                ucb1_learner_item1.update_daily_customers(daily_customers_empirical_means)
                '''
                ts_learner_item1.update_reward_item2(reward_item2)
                ts_learner_item1.update_daily_customers(daily_customers_empirical_means)

                '''
                pulled_arm = ucb1_learner_item1.pull_arm()
                reward = env_item1.round(pulled_arm)
                ucb1_learner_item1.update(pulled_arm, reward)
                '''

                pulled_arm = ts_learner_item1.pull_arm()
                reward = env_item1.round(pulled_arm)
                ts_learner_item1.update(pulled_arm, reward)


                # Non stationary
                # Matching
                conversion_rates_item2_ub = np.zeros([4, 4])
                argmax_index_item2_class1 = np.argmax([sum(SWTS_learner_item2_class1.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * (1-self.discounts)),
                                                       sum(SWTS_learner_item2_class1.get_empirical_means().reshape(3, 4)[0] * prices_item2[1] * (1-self.discounts)),
                                                       sum(SWTS_learner_item2_class1.get_empirical_means().reshape(3, 4)[0] * prices_item2[2] * (1-self.discounts))])
                argmax_index_item2_class2 = np.argmax([sum(SWTS_learner_item2_class2.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * (1-self.discounts)),
                                                       sum(SWTS_learner_item2_class2.get_empirical_means().reshape(3, 4)[0] * prices_item2[1] * (1-self.discounts)),
                                                       sum(SWTS_learner_item2_class2.get_empirical_means().reshape(3, 4)[0] * prices_item2[2] * (1-self.discounts))])
                argmax_index_item2_class3 = np.argmax([sum(SWTS_learner_item2_class3.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * (1-self.discounts)),
                                                       sum(SWTS_learner_item2_class3.get_empirical_means().reshape(3, 4)[0] * prices_item2[1] * (1-self.discounts)),
                                                       sum(SWTS_learner_item2_class3.get_empirical_means().reshape(3, 4)[0] * prices_item2[2] * (1-self.discounts))])
                argmax_index_item2_class4 = np.argmax([sum(SWTS_learner_item2_class4.get_empirical_means().reshape(3, 4)[0] * prices_item2[0] * (1-self.discounts)),
                                                       sum(SWTS_learner_item2_class4.get_empirical_means().reshape(3, 4)[0] * prices_item2[1] * (1-self.discounts)),
                                                       sum(SWTS_learner_item2_class4.get_empirical_means().reshape(3, 4)[0] * prices_item2[2] * (1-self.discounts))])

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
                ucb1_learner_matching.update_daily_customers(daily_customers_empirical_means)

                pulled_arms = ucb1_learner_matching.pull_arm()
                rewards = env_matching.round(pulled_arms)
                ucb1_learner_matching.update(pulled_arms, rewards)

                # Price item 1
                res = np.zeros((3, 4))

                res[pulled_arms[0][0]][pulled_arms[1][0]] = 1
                res[pulled_arms[0][1]][pulled_arms[1][1]] = 1
                res[pulled_arms[0][2]][pulled_arms[1][2]] = 1

                weights = np.zeros((4, 4))
                for i in range(0, 3):
                    for j in range(0, 4):
                        if res[i][j] == 1:
                            weights[i + 1][j] = promo_fractions[i + 1] * sum(daily_customers_empirical_means)

                for j in range(0, 4):
                    weights[0][j] = daily_customers_empirical_means[j] - sum(weights[:, j])

                weights = normalize(weights, 'l1', axis=0)

                reward_item2 = np.zeros(4)
                for i in range(4):
                    reward_item2[i] = sum(prices_item2[majority_voting] * (1-self.discounts) * conversion_rates_item2_ub[:, i] * weights[:, i])

                ts_learner_item1_NS.update_daily_customers(daily_customers_empirical_means)
                ts_learner_item1_NS.update_reward_item2(reward_item2)

                pulled_arm = ts_learner_item1_NS.pull_arm()
                reward = env_item1_NS.round(pulled_arm)
                ts_learner_item1_NS.update(pulled_arm, reward)

            ucb1_rewards_per_experiment_item1.append(ucb1_learner_item1.collected_rewards)
            ts_rewards_per_experiment_item1.append(ts_learner_item1.collected_rewards)
            ts_rewards_per_experiment_item1_NS.append(ts_learner_item1_NS.collected_rewards)

        ts_instantaneous_regret = np.zeros(time_horizon)
        swts_instantaneous_regret = np.zeros(time_horizon)
        optimum_per_round = np.zeros(time_horizon)

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

########################################################################################################################

    def simulation_step_8(self, promo_fractions):
        margins_item1 = self.data.margins_item1
        margins_item2 = self.data.margins_item2

        # TODO create these matrices in Data.py in a non-proportional way (same approach of the non-stationary ones)
        # dim (2,4,5): 2 phases - 4 customers classes - 5 prices
        conversion_rates_item1_NS = np.array([self.data.conversion_rates_item1 + 0.2,
                                              self.data.conversion_rates_item1 - 0.2])
        conversion_rates_item1_NS = np.clip(conversion_rates_item1_NS, 0, 1)

        # dim (2,3,4,4): 2 phases - 3 full prices (40€-50€-60€) - 4 promos - 4 customers classes
        conversion_rates_item2_NS = np.array([self.data.conversion_rates_item2 + 0.2,
                                              self.data.conversion_rates_item2 - 0.2])
        conversion_rates_item2_NS = np.clip(conversion_rates_item2_NS, 0, 1)

        daily_customers = self.data.daily_customers

        # Parameters for the experiments
        n_experiments = 5
        time_horizon = 2000
        n_phases = len(conversion_rates_item1_NS)
        phases_len = time_horizon / n_phases

        # Objective function (in the end, we extract one optimal value per phase)
        objective = np.zeros((n_phases, len(margins_item1), len(margins_item2)))
        for phase in range(n_phases):
            for margin1 in range(len(margins_item1)):
                for margin2 in range(len(margins_item2)):
                    daily_promos = (promo_fractions * sum(daily_customers * conversion_rates_item1_NS[phase, :, margin1])).astype(int)
                    objective[phase][margin1][margin2] = margins_item1[margin1] * (daily_customers * conversion_rates_item1_NS[phase, :, margin1]).sum() + \
                                                         lp.matching_lp(margins_item2[margin2], self.discounts, conversion_rates_item2_NS[phase, margin2], daily_promos,
                                                                        (daily_customers * conversion_rates_item1_NS[phase, :, margin1]).astype(int))[0]

        opt = np.amax(np.amax(objective, axis=2), axis=1)

        # Launching the experiments, using UCB1 with change detection
        # TODO check these parameters and maybe try different ones
        M = 600
        eps = 0.1
        h = np.log(time_horizon) * 2
        alpha = 0.01

        reward_ucb_per_exp = []

        for e in range(n_experiments):
            print("Experiment {}/{} with {} rounds".format(e+1, n_experiments, time_horizon))

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers(np.zeros(4))

            # Environment and learner for the prices of the two items and the matching
            env = Non_Stationary_Environment(margins_item1, margins_item2, conversion_rates_item1_NS,
                                             conversion_rates_item2_NS, daily_customers, self.discounts,
                                             promo_fractions, phases_len)
            ucb_learner = CD_UCB1_Items_Matching(margins_item1, margins_item2, daily_customers, self.discounts, promo_fractions, M, eps, h, alpha)

            reward_ucb_per_round = []

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(daily_customers_sample)

                # Updating the number of customers
                ucb_learner.daily_customers = daily_customers_empirical_means
                env.daily_customers = daily_customers_sample

                # Learning the best prices and matching
                pulled_arm = ucb_learner.pull_arm()
                reward = env.round(pulled_arm)
                ucb_learner.update(pulled_arm[0], reward)

                reward_ucb_per_round.append(reward[2])

            reward_ucb_per_exp.append(reward_ucb_per_round)

        # Computing the optimum per round, according to the optimal value of each phase
        optimum_per_round = np.zeros(time_horizon)
        for i in range(0, n_phases):
            t_index = range(i * int(phases_len), (i + 1) * int(phases_len))
            optimum_per_round[t_index] = opt[i]

        return optimum_per_round, reward_ucb_per_exp, time_horizon
