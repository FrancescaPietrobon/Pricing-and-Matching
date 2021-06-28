from numpy.random import normal

from Data import *
from Environment import *
from Learners.Learner_Customers import *
from Learners.Learner_Conversion_Rates_Item1 import *
from Learners.Learner_Conversion_Rates_Item2 import *
from Learners.Learner_Matching import *
from Learners.UCB1_Item1 import *
from Learners.CD_UCB1_Items_Matching import *
from Learners.TS_Item1 import *
from Learners.SWTS_Items_Matching import *


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

        # Nested loop over the candidate margins of the first and the second item
        # For each pair, it computes the total revenue (objective) and the matching
        objective = np.zeros((len(margins_item1), len(margins_item2)))
        matching = np.zeros((len(margins_item1), len(margins_item2), 4, 4), dtype=int)
        for margin1 in range(len(margins_item1)):
            for margin2 in range(len(margins_item2)):
                daily_promos = (promo_fractions * sum(daily_customers * conversion_rates_item1[:, margin1])).astype(int)
                revenue_item2, matching[margin1, margin2] = lp.matching_lp(margins_item2[margin2], self.discounts,
                                                                           conversion_rates_item2[margin2], daily_promos,
                                                                           (daily_customers * conversion_rates_item1[:, margin1]).astype(int))
                objective[margin1, margin2] = margins_item1[margin1] * (daily_customers * conversion_rates_item1[:, margin1]).sum() + revenue_item2

        # Extracting the prices that maximize the total revenue (objective)
        idx_best_margin_item1, idx_best_margin_item2 = np.unravel_index(np.argmax(objective), objective.shape)
        best_price_item1 = self.data.prices_item1[idx_best_margin_item1]
        best_price_item2 = self.data.prices_item2[idx_best_margin_item2]

        # Returning the maximum objective value obtained, along with the corresponding best prices for the items
        # and the best promo-customer class matching matrix
        return objective[idx_best_margin_item1, idx_best_margin_item2], best_price_item1, best_price_item2, matching[idx_best_margin_item1, idx_best_margin_item2]

########################################################################################################################

    def simulation_step_3(self, promo_fractions):
        # Taking the optimal known values from Step 1
        opt, _, selected_price_item2, matching = self.simulation_step_1(promo_fractions)
        conversion_rates_item1 = self.data.conversion_rates_item1
        conversion_rates_item2 = self.data.conversion_rates_item2[list(self.data.prices_item2).index(selected_price_item2)]
        selected_margin_item2 = self.data.margins_item2[list(self.data.prices_item2).index(selected_price_item2)]
        weights = normalize(matching, norm='l1', axis=0)

        margins_item1 = self.data.margins_item1
        daily_customers = self.data.daily_customers

        # Launching the experiments, using both UCB1 and Thompson Sampling to learn the price of the first item
        n_experiments = 10
        time_horizon = 365
        n_arms = len(margins_item1)
        ucb1_rewards_per_experiment = []
        ts_rewards_per_experiment = []

        for e in range(n_experiments):
            print("Experiment {}/{} with {} rounds".format(e+1, n_experiments, time_horizon))

            env = Environment_Single_Price(margins_item1, selected_margin_item2, conversion_rates_item1,
                                           conversion_rates_item2, weights, daily_customers, self.discounts)
            ucb1_learner = UCB1_Item1(n_arms, margins_item1, selected_margin_item2, conversion_rates_item2, weights,
                                      daily_customers, self.discounts)
            ts_learner = TS_Item1(n_arms, margins_item1, selected_margin_item2, conversion_rates_item2, weights,
                                  daily_customers, self.discounts)

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
        # Taking the optimal known values from Step 1
        opt, _, selected_price_item2, weights = self.simulation_step_1(promo_fractions)
        conversion_rates_item1 = self.data.conversion_rates_item1
        conversion_rates_item2 = self.data.conversion_rates_item2[list(self.data.prices_item2).index(selected_price_item2)]
        selected_margin_item2 = self.data.margins_item2[list(self.data.prices_item2).index(selected_price_item2)]
        weights = normalize(weights, norm='l1', axis=0)

        margins_item1 = self.data.margins_item1
        daily_customers = self.data.daily_customers

        # Launching the experiments, using both UCB1 and Thompson Sampling to learn the price of the first item
        # This time, the number of customers and the conversion rates for the second item are not known
        n_experiments = 10
        time_horizon = 365
        n_arms = len(margins_item1)
        ucb1_rewards_per_experiment = []
        ts_rewards_per_experiment = []

        for e in range(n_experiments):
            print("Experiment {}/{} with {} rounds".format(e+1, n_experiments, time_horizon))

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers(np.zeros(4))

            # Learners for the mean of the conversion rates of the second item
            learner_conversion_rates_item2_ucb1 = Learner_Conversion_Rates_item2()
            learner_conversion_rates_item2_ts = Learner_Conversion_Rates_item2()

            # Environment and learners (UCB1 and Thompson Sampling) for the price of the first item
            env = Environment_Single_Price(margins_item1, selected_margin_item2, conversion_rates_item1,
                                           conversion_rates_item2, weights, daily_customers, self.discounts)
            ucb1_learner = UCB1_Item1(n_arms, margins_item1, selected_margin_item2, conversion_rates_item2, weights,
                                      daily_customers, self.discounts)
            ts_learner = TS_Item1(n_arms, margins_item1, selected_margin_item2, conversion_rates_item2, weights,
                                  daily_customers, self.discounts)

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

                # Learning the conversion rates for the second item, updating them in the UCB1 learner
                conversion_rates_item2_means = learner_conversion_rates_item2_ucb1.update_conversion_rates(conversion_rates_item2_sample)
                ucb1_learner.conversion_rates_item2 = conversion_rates_item2_means

                # Thompson Sampling Learner
                pulled_arm = ts_learner.pull_arm()
                reward, conversion_rates_item2_sample, revenue = env.round(pulled_arm)
                ts_learner.update(pulled_arm, reward, revenue)

                # Learning the conversion rates for the second item, updating them in the Thompson Sampling learner
                conversion_rates_item2_means = learner_conversion_rates_item2_ts.update_conversion_rates(conversion_rates_item2_sample)
                ts_learner.conversion_rates_item2 = conversion_rates_item2_means

            ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
            ts_rewards_per_experiment.append(ts_learner.collected_rewards)

        return opt, ucb1_rewards_per_experiment, ts_rewards_per_experiment, time_horizon

########################################################################################################################

    def simulation_step_5(self, promo_fractions):
        # Taking the optimal known values from Step 1
        opt, selected_price_item1, selected_price_item2, _ = self.simulation_step_1(promo_fractions)
        conversion_rates_item1 = self.data.conversion_rates_item1[:, list(self.data.prices_item1).index(selected_price_item1)]
        selected_margin_item1 = self.data.margins_item1[list(self.data.prices_item1).index(selected_price_item1)]
        conversion_rates_item2 = self.data.conversion_rates_item2[list(self.data.prices_item2).index(selected_price_item2)]
        selected_margin_item2 = self.data.margins_item2[list(self.data.prices_item2).index(selected_price_item2)]

        daily_customers = self.data.daily_customers

        # Launching the experiments, using the LP to learn the promo-customer class matching
        n_experiments = 10
        time_horizon = 365
        reward_matching_per_exp = []

        for e in range(n_experiments):
            print("Experiment {}/{} with {} rounds".format(e+1, n_experiments, time_horizon))

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers(np.zeros(4))

            # Learner for the mean of the conversion rates of the first item
            learner_conversion_rates_item1 = Learner_Conversion_Rates_item1()

            # Learner for the mean of the conversion rates of the second item
            learner_conversion_rates_item2 = Learner_Conversion_Rates_item2()

            # Environment and learner for the matching between promos and customer classes
            env = Environment_Matching(selected_margin_item1, selected_margin_item2, conversion_rates_item1,
                                       conversion_rates_item2, daily_customers, self.discounts)
            matching_learner = Matching(selected_margin_item2, daily_customers, self.discounts, promo_fractions)

            rew_matching_per_round = []

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(daily_customers_sample)

                # Updating the daily customers in the learner and in the environment
                matching_learner.daily_customers = daily_customers_empirical_means
                env.daily_customers = daily_customers_sample

                # Computing the matching weights matrix (normalized)
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
        # Taking the optimal objective value from Step 1
        opt, _, _, _ = self.simulation_step_1(promo_fractions)

        margins_item1 = self.data.margins_item1
        margins_item2 = self.data.margins_item2
        conversion_rates_item1 = self.data.conversion_rates_item1
        conversion_rates_item2 = self.data.conversion_rates_item2
        daily_customers = self.data.daily_customers

        # Launching the experiments, using both UCB1 and TS to learn the prices for the two items and the matching
        n_experiments = 10
        time_horizon = 365
        reward_ucb_per_exp = []
        reward_ts_per_exp = []

        for e in range(n_experiments):
            print("Experiment {}/{} with {} rounds".format(e+1, n_experiments, time_horizon))

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers(np.zeros(4))

            # Environment and learners for the prices of the two items and the matching
            env = Environment_Double_Prices_Matching(margins_item1, margins_item2, conversion_rates_item1,
                                                     conversion_rates_item2, daily_customers, self.discounts, promo_fractions)
            ucb_learner = UCB1_Items_Matching(margins_item1, margins_item2, daily_customers, self.discounts, promo_fractions)
            ts_learner = TS_Items_Matching(margins_item1, margins_item2, daily_customers, self.discounts, promo_fractions)

            reward_ucb_per_round = []
            reward_ts_per_round = []

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(daily_customers_sample)

                # Updating the daily customers in the learners and in the environment
                ucb_learner.daily_customers = daily_customers_empirical_means
                ts_learner.daily_customers = daily_customers_empirical_means
                env.daily_customers = daily_customers_sample

                # Learning the best prices and matching (UCB1)
                pulled_arm = ucb_learner.pull_arm()
                reward = env.round(pulled_arm)
                ucb_learner.update(pulled_arm[0], reward)

                reward_ucb_per_round.append(reward[2])

                # Learning the best prices and matching (Thompson Sampling)
                pulled_arm = ts_learner.pull_arm()
                reward = env.round(pulled_arm)
                ts_learner.update(pulled_arm[0], reward)

                reward_ts_per_round.append(reward[2])

            reward_ucb_per_exp.append(reward_ucb_per_round)
            reward_ts_per_exp.append(reward_ts_per_round)

        return opt, reward_ucb_per_exp, reward_ts_per_exp, time_horizon

########################################################################################################################

    def simulation_step_7(self, promo_fractions):
        margins_item1 = self.data.margins_item1
        margins_item2 = self.data.margins_item2
        conversion_rates_item1_NS = self.data.conversion_rates_item1_NS
        conversion_rates_item2_NS = self.data.conversion_rates_item2_NS
        daily_customers = self.data.daily_customers

        # Launching the experiments, using SW-TS to learn the prices of the two items and the matching
        n_phases = len(conversion_rates_item1_NS)
        n_experiments = 10
        time_horizon = 365 * n_phases
        phases_len = time_horizon / n_phases
        window_size = int(np.sqrt(time_horizon))
        reward_swts_per_exp = []

        for e in range(n_experiments):
            print("Experiment {}/{} with {} rounds".format(e + 1, n_experiments, time_horizon))

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers(np.zeros(4))

            # Environment and learner for the prices of the two items and the matching
            env = Non_Stationary_Environment(margins_item1, margins_item2, conversion_rates_item1_NS, conversion_rates_item2_NS,
                                             daily_customers, self.discounts, promo_fractions, phases_len)
            swts_learner = SWTS_Items_Matching(window_size, margins_item1, margins_item2, daily_customers, self.discounts, promo_fractions)

            reward_swts_per_round = []

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(daily_customers_sample)

                # Updating the daily customers in the learner and in the environment
                swts_learner.daily_customers = daily_customers_empirical_means
                env.daily_customers = daily_customers_sample

                # Learning the best prices and matching
                pulled_arm = swts_learner.pull_arm()
                reward = env.round(pulled_arm)
                swts_learner.update(pulled_arm[0], reward)

                reward_swts_per_round.append(reward[2])

            reward_swts_per_exp.append(reward_swts_per_round)

        # Objective function (total revenue, depending on the phase and the margins of the two items)
        objective = np.zeros((n_phases, len(margins_item1), len(margins_item2)))
        for phase in range(n_phases):
            for margin1 in range(len(margins_item1)):
                for margin2 in range(len(margins_item2)):
                    daily_promos = (promo_fractions * sum(daily_customers * conversion_rates_item1_NS[phase, :, margin1])).astype(int)
                    objective[phase][margin1][margin2] = margins_item1[margin1] * (daily_customers * conversion_rates_item1_NS[phase, :, margin1]).sum() + \
                                                         lp.matching_lp(margins_item2[margin2], self.discounts, conversion_rates_item2_NS[phase, margin2], daily_promos,
                                                                        (daily_customers * conversion_rates_item1_NS[phase, :, margin1]).astype(int))[0]

        # Extracting the maximum objective value for each phase
        opt = np.amax(np.amax(objective, axis=2), axis=1)

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
        conversion_rates_item1_NS = self.data.conversion_rates_item1_NS
        conversion_rates_item2_NS = self.data.conversion_rates_item2_NS
        daily_customers = self.data.daily_customers

        # Launching the experiments, using CD-UCB1 (CUMSUM) to learn the prices of the two items and the matching
        n_phases = len(conversion_rates_item1_NS)
        n_experiments = 10
        time_horizon = 365 * n_phases
        phases_len = time_horizon / n_phases

        M = int(time_horizon / 3)
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
            ucb_learner = CD_UCB1_Items_Matching(margins_item1, margins_item2, daily_customers, self.discounts,
                                                 promo_fractions, M, eps, h, alpha)

            reward_ucb_per_round = []

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(daily_customers_sample)

                # Updating the daily customers in the learner and in the environment
                ucb_learner.daily_customers = daily_customers_empirical_means
                env.daily_customers = daily_customers_sample

                # Learning the best prices and matching
                pulled_arm = ucb_learner.pull_arm()
                reward = env.round(pulled_arm)
                ucb_learner.update(pulled_arm[0], reward)

                reward_ucb_per_round.append(reward[2])

            reward_ucb_per_exp.append(reward_ucb_per_round)

        # Objective function (total revenue, depending on the phase and the margins of the two items)
        objective = np.zeros((n_phases, len(margins_item1), len(margins_item2)))
        for phase in range(n_phases):
            for margin1 in range(len(margins_item1)):
                for margin2 in range(len(margins_item2)):
                    daily_promos = (promo_fractions * sum(daily_customers * conversion_rates_item1_NS[phase, :, margin1])).astype(int)
                    objective[phase][margin1][margin2] = margins_item1[margin1] * (daily_customers * conversion_rates_item1_NS[phase, :, margin1]).sum() + \
                                                         lp.matching_lp(margins_item2[margin2], self.discounts, conversion_rates_item2_NS[phase, margin2], daily_promos,
                                                                        (daily_customers * conversion_rates_item1_NS[phase, :, margin1]).astype(int))[0]

        # Extracting the maximum objective value for each phase
        opt = np.amax(np.amax(objective, axis=2), axis=1)

        # Computing the optimum per round, according to the optimal value of each phase
        optimum_per_round = np.zeros(time_horizon)
        for i in range(0, n_phases):
            t_index = range(i * int(phases_len), (i + 1) * int(phases_len))
            optimum_per_round[t_index] = opt[i]
        optimum_per_round[-1] = optimum_per_round[-2]          # To correctly set the last element due to approximations

        return optimum_per_round, reward_ucb_per_exp, time_horizon
