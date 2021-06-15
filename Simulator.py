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
from Learners.SWTS_Items_Matching import *
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
        n_experiments = 10
        time_horizon = 365
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
        n_experiments = 10
        time_horizon = 365
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
        n_experiments = 10
        time_horizon = 365
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
        n_experiments = 10
        time_horizon = 365

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

    def simulation_step_7(self, promo_fractions):
        margins_item1 = self.data.margins_item1
        margins_item2 = self.data.margins_item2

        conversion_rates_item1_NS = np.array([self.data.conversion_rates_item1 + 0.2,
                                              self.data.conversion_rates_item1,
                                              self.data.conversion_rates_item1 - 0.1])
        conversion_rates_item1_NS = np.clip(conversion_rates_item1_NS, 0, 1)

        conversion_rates_item2_NS = np.array([self.data.conversion_rates_item2 + 0.2,
                                              self.data.conversion_rates_item2,
                                              self.data.conversion_rates_item2 - 0.1])
        conversion_rates_item2_NS = np.clip(conversion_rates_item2_NS, 0, 1)

        daily_customers = self.data.daily_customers

        # Parameters for the experiments
        n_experiments = 10
        time_horizon = 365
        n_phases = len(conversion_rates_item1_NS)
        phases_len = time_horizon / n_phases
        window_size = int(np.sqrt(time_horizon))

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

        reward_swts_per_exp = []

        for e in range(n_experiments):
            print("Experiment {}/{} with {} rounds".format(e + 1, n_experiments, time_horizon))

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers(np.zeros(4))

            # Environment and learner for the prices of the two items and the matching
            env = Non_Stationary_Environment(margins_item1, margins_item2, conversion_rates_item1_NS,
                                             conversion_rates_item2_NS, daily_customers, self.discounts,
                                             promo_fractions, phases_len)
            swts_learner = SWTS_Items_Matching(window_size, margins_item1, margins_item2, daily_customers, self.discounts,
                                              promo_fractions)

            reward_swts_per_round = []

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(daily_customers_sample)

                # Updating the number of customers
                swts_learner.daily_customers = daily_customers_empirical_means
                env.daily_customers = daily_customers_sample

                # Learning the best prices and matching
                pulled_arm = swts_learner.pull_arm()
                reward = env.round(pulled_arm)
                swts_learner.update(pulled_arm[0], reward)

                reward_swts_per_round.append(reward[2])

            reward_swts_per_exp.append(reward_swts_per_round)

        # Computing the optimum per round, according to the optimal value of each phase
        optimum_per_round = np.zeros(time_horizon)
        for i in range(0, n_phases):
            t_index = range(i * int(phases_len), (i + 1) * int(phases_len))
            optimum_per_round[t_index] = opt[i]
        optimum_per_round[-1] = optimum_per_round[-2]          # To correctly set the last element due to approximations

        return optimum_per_round, reward_swts_per_exp, time_horizon

########################################################################################################################

    def simulation_step_8(self, promo_fractions):
        margins_item1 = self.data.margins_item1
        margins_item2 = self.data.margins_item2

        conversion_rates_item1_NS = np.array([self.data.conversion_rates_item1 + 0.2,
                                              self.data.conversion_rates_item1,
                                              self.data.conversion_rates_item1 - 0.1])
        conversion_rates_item1_NS = np.clip(conversion_rates_item1_NS, 0, 1)

        conversion_rates_item2_NS = np.array([self.data.conversion_rates_item2 + 0.2,
                                              self.data.conversion_rates_item2,
                                              self.data.conversion_rates_item2 - 0.1])
        conversion_rates_item2_NS = np.clip(conversion_rates_item2_NS, 0, 1)

        daily_customers = self.data.daily_customers

        # Parameters for the experiments
        n_experiments = 10
        time_horizon = 5000
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
        M = 2000
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
        optimum_per_round[-1] = optimum_per_round[-2]          # To correctly set the last element due to approximations

        return optimum_per_round, reward_ucb_per_exp, time_horizon
