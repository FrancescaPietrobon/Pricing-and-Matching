import matplotlib.pyplot as plt
from numpy.random import normal
from scipy import stats
from sklearn.preprocessing import normalize

from CUMSUM_UCB1_item1 import CUMSUM_UCB1_item1
from CUMSUM_UCB1_item2 import CUMSUM_UCB1_item2
from CUMSUM_UCB_Matching import CUMSUM_UCB_Matching
from Data import *
from Environment import *
from Item import *
import matching_lp as lp
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

########################################################################################################################

    def simulation_step_1(self, promo_fractions):
        # Daily number of customers per class
        daily_customers = self.data.daily_customers

        # Prices item 2
        prices_item2 = self.data.prices_item2

        # Number of promo codes available daily (fixed fraction of the daily number of customers)
        daily_promos = (promo_fractions * sum(daily_customers)).astype(int)

        # Probability that a customer of a class buys the second item given that he bought the first and has a promo
        # 4x4 matrix -> rows: promo (P0, P1, P2, P3); columns: customer class (class1, class2, class3, class4)
        # TODO create different conversion_rates_item21 matrices (not proportional changes)
        prob_buy_item2 = np.array([self.data.conversion_rates_item21 + 0.1,
                                   self.data.conversion_rates_item21,
                                   self.data.conversion_rates_item21 - 0.1])
        prob_buy_item2 = np.clip(prob_buy_item2, 0, 1)

        result = np.zeros((len(prices_item2), 4, 4))

        for i in range(len(prices_item2)):
            result[i] = lp.matching_lp(prices_item2[i], self.discounts, prob_buy_item2[i], daily_promos, daily_customers)[1]

        return result, prices_item2

########################################################################################################################

    def simulation_step_3(self, promo_fractions):
        # Number of arms (computed as np.ceil((T * np.log(T))**(1/4)).astype(int))
        n_arms = 9

        # Candidate prices for item 1 (one per arm)
        prices_item1 = self.data.prices_item1

        # Conversion rates for item 1 (4x9 matrix: one per arm per customer class)
        conversion_rates_item1 = self.data.conversion_rates_item1

        # Conversion rates for item 2 given item 1 and promo (row: promo; column: customer class)
        conversion_rates_item2 = self.data.conversion_rates_item21

        # Number of daily customers (one per class)
        daily_customers = self.data.daily_customers

        # Promo assigment weights (row: promo; column: customer class) - Taken by step 1
        weights = normalize(self.simulation_step_1(promo_fractions)[1], 'l1', axis=0)

        # Objective array (one element per arm)
        objective = np.zeros(n_arms)
        for i in range(n_arms):
            objective[i] = sum(prices_item1[i] * daily_customers * conversion_rates_item1[:, i] +
                               self.item2.price * daily_customers * conversion_rates_item1[:, i] *
                               (np.dot(1-self.discounts, conversion_rates_item2 * weights)))

        # Optimal objective value, useful to compute the regret later
        opt = max(objective)

        # Reward obtained when buying item 2 (one element per customer class)
        reward_item2 = np.zeros(4)
        for i in range(4):
            reward_item2[i] = sum(self.item2.price * (1-self.discounts) * conversion_rates_item2[:, i] * weights[:, i])

        # Launching the experiments, using both UCB1 and Thompson Sampling to learn the price of item 1
        n_experiments = 100
        time_horizon = 1000
        ucb1_rewards_per_experiment = []
        ts_rewards_per_experiment = []

        for e in range(n_experiments):
            print("Experiment ", e+1, "/", n_experiments)

            env = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)
            ucb1_learner = UCB1_item1(n_arms, daily_customers, prices_item1, reward_item2)
            ts_learner = TS_Learner_item1(n_arms, daily_customers, prices_item1, reward_item2)

            for t in range(time_horizon):
                # TODO generate customer after customer (maybe in the environment) using various bernoulli (only item 1 and, inside, item 2 with different promos) incrementing rewad_item2 for each selling
                #  Then, in the learner delete daily_customers and use the "reward" as the actual number of customer who bought item 2 (in TS we have 1-reward -> total customers - actual buyers)
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

        return [opt, ucb1_rewards_per_experiment, ts_rewards_per_experiment]

########################################################################################################################

    def simulation_step_4(self, promo_fractions):
        # Number of arms (computed as np.ceil((T * np.log(T))**(1/4)).astype(int))
        n_arms = 9

        # Candidate prices for item 1 (one per arm)
        prices_item1 = self.data.prices_item1

        # Conversion rates for item 1 (4x9 matrix: one per arm per customer class)
        conversion_rates_item1 = self.data.conversion_rates_item1

        # Conversion rates for item 2 given item 1 and promo (row: promo; column: customer class)
        conversion_rates_item2 = self.data.conversion_rates_item21

        # Number of daily customers (one per class)
        daily_customers = self.data.daily_customers

        # Promo assigment weights (row: promo; column: customer class) - Taken by step 1
        weights = normalize(self.simulation_step_1(promo_fractions)[1], 'l1', axis=0)

        # Objective array (one element per arm)
        objective = np.zeros(n_arms)
        for i in range(n_arms):
            objective[i] = sum(prices_item1[i] * daily_customers * conversion_rates_item1[:, i] +
                               self.item2.price * daily_customers * conversion_rates_item1[:, i] *
                               (np.dot(1-self.discounts, conversion_rates_item2 * weights)))

        # Optimal objective value, useful to compute the regret later
        opt = max(objective)

        # Launching the experiments, using UCB1 to learn the conversion rates of item 2
        # and both UCB1 and Thompson Sampling to learn the price of item 1
        n_experiments = 100
        time_horizon = 1000
        ucb1_rewards_per_experiment_item1 = []
        ts_rewards_per_experiment_item1 = []

        for e in range(n_experiments):
            print("Experiment ", e+1, "/", n_experiments)

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers()
            daily_customers_empirical_means = np.zeros(4)

            # Environments and learners for the conversion rates of item 2 (one per customer class)
            envs_item2 = np.array([Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 0]),
                                   Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 1]),
                                   Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 2]),
                                   Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 3])])

            ucb1_learners_item2 = np.array([UCB1_item2(n_arms=4),
                                            UCB1_item2(n_arms=4),
                                            UCB1_item2(n_arms=4),
                                            UCB1_item2(n_arms=4)])

            # Environment and learners (UCB1 and Thompson Sampling) for the price of item 1
            env_item1 = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)
            ucb1_learner_item1 = UCB1_item1(n_arms, daily_customers, prices_item1, reward_item2=np.zeros(4))
            ts_learner_item1 = TS_Learner_item1(n_arms, daily_customers, prices_item1, reward_item2=np.zeros(4))

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(
                    empirical_means=daily_customers_empirical_means, sample=daily_customers_sample)

                # Learning the conversion rates for item 2
                for learner, env in zip(ucb1_learners_item2, envs_item2):
                    pulled_arm = learner.pull_arm()
                    reward = env.round(pulled_arm)
                    learner.update(pulled_arm, reward)

                # Retrieving the conversion rates in the usual 4x4 matrix (row: promo; column: customer class)
                conversion_rates_item2_em = np.zeros((4, 4))
                for i in range(4):
                    conversion_rates_item2_em[:, i] = ucb1_learners_item2[i].empirical_means

                # Computing the reward obtained when buying item 2 (one element per customer class)
                reward_item2 = np.zeros(4)
                for i in range(4):
                    reward_item2[i] = sum(self.item2.price * (1-self.discounts) *
                                          conversion_rates_item2_em[:, i] * weights[:, i])

                # Updating the daily customers and the reward of item 2 in the learners for the price of item 1
                ucb1_learner_item1.daily_customers = daily_customers_empirical_means
                ts_learner_item1.daily_customers = daily_customers_empirical_means
                ucb1_learner_item1.reward_item2 = reward_item2
                ts_learner_item1.reward_item2 = reward_item2

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

        return [opt, ucb1_rewards_per_experiment_item1, ts_rewards_per_experiment_item1]

########################################################################################################################

    def simulation_step_5(self, promo_fractions):
        # Conversion rates for item 2 given item 1 and promo (row: promo; column: customer class)
        conversion_rates_item2 = self.data.conversion_rates_item21

        # Number of daily customers (one per class)
        daily_customers = self.data.daily_customers

        # Objective matrix (row: promo; column: customer class)
        # Assumption: the number of promos of every type of promo is smaller than the number of customers of every class
        objective = np.zeros((3, 4))
        for i in range(3):
            objective[i][:] = (self.item2.price * daily_customers * conversion_rates_item2[i+1][:] *
                               (1-self.discounts[i+1]) * promo_fractions[i+1])

        # Optimal objective value, useful to compute the regret later
        opt = linear_sum_assignment(-objective)

        # Launching the experiments, using UCB1 to learn the conversion rates of item 2 and the matching
        n_experiments = 100
        time_horizon = 5000
        regret_ucb = np.zeros((n_experiments, time_horizon))
        reward_ucb = []
        for e in range(n_experiments):
            print("Experiment ", e+1, "/", n_experiments)

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers()
            daily_customers_empirical_means = np.zeros(4)

            # Environments and learners for the conversion rates of item 2 (one per customer class)
            envs_item2 = np.array([Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 0]),
                                   Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 1]),
                                   Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 2]),
                                   Environment_First(n_arms=4, probabilities=conversion_rates_item2[:, 3])])

            ucb1_learners_item2 = np.array([UCB1_item2(n_arms=4),
                                            UCB1_item2(n_arms=4),
                                            UCB1_item2(n_arms=4),
                                            UCB1_item2(n_arms=4)])

            # Environment and learner for the matching between promos and customer classes
            probabilities = np.zeros((3, 4))
            env_matching = Environment_First(probabilities.size, probabilities)
            learner_matching = UCB_Matching(probabilities.size, *probabilities.shape, self.item2.price,
                                            daily_customers, 1-self.discounts, promo_fractions)

            rew_UCB = []
            opt_rew = []

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(
                    empirical_means=daily_customers_empirical_means, sample=daily_customers_sample)

                # Learning the conversion rates for item 2
                for learner, env in zip(ucb1_learners_item2, envs_item2):
                    pulled_arm = learner.pull_arm()
                    reward = env.round(pulled_arm)
                    learner.update(pulled_arm, reward)

                # Retrieving the conversion rates in a 3x4 matrix (row: promo; column: customer class)
                # As always, we do not consider promo P0 for the matching, since it corresponds to "no discount"
                for i in range(4):
                    probabilities[:, i] = ucb1_learners_item2[i].empirical_means[1:]

                # Updating the daily customers in the learner and the probabilities vector in the environment
                learner_matching.daily_customers = daily_customers_empirical_means
                env_matching.probabilities = probabilities

                # Learning the matching using UCB1
                pulled_arms = learner_matching.pull_arm()
                rewards = env_matching.round(pulled_arms)
                learner_matching.update(pulled_arms, rewards)

                # Why customers[pulled_arms[1]]?
                # Because pulled_arms is a tuple [[a, b, c],[x, y, z]] which specifies how to assign promos a, b, c
                # to customer classes x, y, z. Therefore, for the reward computation we consider classes x, y, z
                rew_UCB.append((self.item2.price * (1-self.discounts[1:]) * promo_fractions[1:] * rewards *
                                daily_customers_empirical_means[pulled_arms[1]]).sum())
                opt_rew.append(objective[opt].sum())

            regret_ucb[e, :] = np.cumsum(opt_rew) - np.cumsum(rew_UCB)
            reward_ucb.append(rew_UCB)

        return [regret_ucb, reward_ucb]

########################################################################################################################

    def simulation_step_6(self, promo_fractions):
        # Number of arms for pricing item 1
        n_arms = 9

        # Candidate prices for item 1 and item 2 (one per arm)
        prices_item1 = self.data.prices_item1
        prices_item2 = self.data.prices_item2

        # Conversion rates for item 1 (one per arm)
        conversion_rates_item1 = self.data.conversion_rates_item1

        # Conversion rates for item 2 (one 4x4 matrix per arm)
        conversion_rates_item2 = np.array([self.data.conversion_rates_item21 + 0.1,
                                           self.data.conversion_rates_item21,
                                           self.data.conversion_rates_item21 - 0.1])
        conversion_rates_item2 = np.clip(conversion_rates_item2, 0, 1)

        # Number of daily customers per class
        daily_customers = self.data.daily_customers

        # Promo assigment weights (one matrix for each candidate price for item 2; row: promo; column: customer class)
        daily_promos = (promo_fractions * sum(daily_customers)).astype(int)
        weights = np.zeros((3, 4, 4))
        for k in range(3):
            weights[k] = normalize(lp.matching_lp(prices_item2[k], self.discounts, conversion_rates_item2[k],
                                                  daily_promos, daily_customers)[1], 'l1', axis=0)

        # Objective array (one element for every combination of candidate prices for item 1 and item 2)
        objective = np.zeros((n_arms, 3))
        for i in range(n_arms):
            for k in range(3):
                objective[i, k] = sum(prices_item1[i] * daily_customers * conversion_rates_item1[:, i] +
                                      prices_item2[k] * daily_customers * conversion_rates_item1[:, i] *
                                      (np.dot(1 - self.discounts, conversion_rates_item2[k] * weights[k])))

        # Storing the optimal objective value to compute the regret later
        opt = np.max(objective)

        # Launching the experiments
        n_experiments = 20
        time_horizon = 5000

        ucb1_rewards_per_experiment_item1 = []
        ts_rewards_per_experiment_item1 = []

        for e in range(n_experiments):
            print("Experiment ", e+1, "/", n_experiments)

            # Environment and learner for the number of daily customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers()
            daily_customers_empirical_means = np.zeros(4)

            # Environments and learners for the conversion rates of item 2 (one per customer class)
            envs_item2 = np.array([Environment_First(n_arms=12, probabilities=conversion_rates_item2[:, :, 0].flatten()),
                                   Environment_First(n_arms=12, probabilities=conversion_rates_item2[:, :, 1].flatten()),
                                   Environment_First(n_arms=12, probabilities=conversion_rates_item2[:, :, 2].flatten()),
                                   Environment_First(n_arms=12, probabilities=conversion_rates_item2[:, :, 3].flatten())])

            ucb1_learners_item2 = np.array([UCB1_item2(n_arms=12),
                                            UCB1_item2(n_arms=12),
                                            UCB1_item2(n_arms=12),
                                            UCB1_item2(n_arms=12)])

            # Environment and learner for the matching between promos and customer classes
            probabilities = np.zeros((3, 4))
            ucb1_learner_matching = UCB_Matching(probabilities.size, *probabilities.shape, price=0,daily_customers=daily_customers, discounts=1-self.discounts, p_frac=promo_fractions)
            env_matching = Environment_First(probabilities.size, probabilities)

            # Environment and learner for the price of item 1
            env_item1 = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)
            ucb1_learner_item1 = UCB1_item1(n_arms=n_arms, daily_customers=daily_customers, prices=prices_item1, reward_item2=np.zeros(4))
            ts_learner_item1 = TS_Learner_item1(n_arms=n_arms, daily_customers=daily_customers, prices=prices_item1, reward_item2=np.zeros(4))


            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(
                    empirical_means=daily_customers_empirical_means, sample=daily_customers_sample)

                # Learning the conversion rates for item 2
                for learner, env in zip(ucb1_learners_item2, envs_item2):
                    pulled_arm = learner.pull_arm()
                    reward = env.round(pulled_arm)
                    learner.update(pulled_arm, reward)

                # Taking the best price for item 2 for each class
                # Each "empirical_means" is reshaped into a 3x4 matrix (3 prices for item 2, 4 conversion rates)
                indices_best_price_item2 = np.zeros(4)
                for i in range(4):
                    indices_best_price_item2[i] = np.argmax(
                        np.dot(ucb1_learners_item2[i].empirical_means.reshape(3, 4), (1 - self.discounts)) * prices_item2,
                        axis=0)

                # Computing the most common price selected, since we want the same price for all the customer classes
                best_price_item2 = stats.mode(indices_best_price_item2)[0][0].astype(int)

                # Constructing the usual 4x4 matrix of conversion rates for item 2
                conversion_rates_item2_em = np.zeros((4, 4))
                for i in range(4):
                    conversion_rates_item2_em[:, i] = ucb1_learners_item2[i].empirical_means.reshape(3, 4)[best_price_item2]

                # Updating the probabilities vector in the environment used by UCB_matching
                # as well as the best price for item 2 that has been selected, and the number of customers learned
                env_matching.probabilities = conversion_rates_item2_em[1:, :]
                ucb1_learner_matching.price = prices_item2[best_price_item2]
                ucb1_learner_matching.daily_customers = daily_customers_empirical_means

                # Learning the best promo-customer class matching
                pulled_arms = ucb1_learner_matching.pull_arm()
                rewards = env_matching.round(pulled_arms)
                ucb1_learner_matching.update(pulled_arms, rewards)

                # Given the resulting pulled arms [[x, y, z], [a, b, c]], we reconstruct the usual 4x4 matrix
                # In the cells selected by the matching above, we give the promo (as a fraction of the customers)
                # Notice that the "+1" on the rows is required since the matching did not consider promo P0
                weights = np.zeros((4, 4))
                for i in range(0, 3):
                    for j in range(0, 4):
                        weights[pulled_arms[0][i] + 1, pulled_arms[1][i]] = promo_fractions[i + 1] * sum(
                            daily_customers_empirical_means)

                # Otherwise, as always, we give promo P0 to the remaining customers
                for j in range(0, 4):
                    weights[0, j] = daily_customers_empirical_means[j] - sum(weights[:, j])

                # Normalizing the weights matrix to have proper values between 0 and 1
                weights = normalize(weights, 'l1', axis=0)

                # Computing the reward obtained when buying item 2 (one element per customer class)
                reward_item2 = np.zeros(4)
                for i in range(4):
                    reward_item2[i] = sum(prices_item2[best_price_item2] * (1 - self.discounts) *
                                          conversion_rates_item2_em[:, i] * weights[:, i])

                # Updating the number of customers and the reward given by item 2 in the learners for price of item 1
                ucb1_learner_item1.reward_item2 = reward_item2
                ucb1_learner_item1.daily_customers = daily_customers_empirical_means
                ts_learner_item1.reward_item2 = reward_item2
                ts_learner_item1.daily_customers = daily_customers_empirical_means

                # Learning the best price for item 1 (UCB1)
                pulled_arm = ucb1_learner_item1.pull_arm()
                reward = env_item1.round(pulled_arm)
                ucb1_learner_item1.update(pulled_arm, reward)

                # Learning the best price for item 1 (Thompson Sampling)
                pulled_arm = ts_learner_item1.pull_arm()
                reward = env_item1.round(pulled_arm)
                ts_learner_item1.update(pulled_arm, reward)

            ucb1_rewards_per_experiment_item1.append(ucb1_learner_item1.collected_rewards)
            ts_rewards_per_experiment_item1.append(ts_learner_item1.collected_rewards)

        return [opt, ucb1_rewards_per_experiment_item1, ts_rewards_per_experiment_item1]

########################################################################################################################
    # TODO clean code
    def simulation_step_7(self, promo_fractions):
        # Number of arms for pricing item 1
        n_arms = 9

        # Candidate prices for item 1 and item 2 (one per arm)
        prices_item1 = self.data.prices_item1
        prices_item2 = self.data.prices_item2

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
            env_matching = Environment_First(probabilities.size, probabilities)

            # env_item1 = Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1)
            env_item1 = Non_Stationary_Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1_NS, horizon=time_horizon)
            env_item1_NS = Non_Stationary_Environment_Third(n_arms=n_arms, probabilities=conversion_rates_item1_NS, horizon=time_horizon)

            ucb1_learner_item1 = UCB1_item1(n_arms=n_arms, daily_customers=daily_customers, prices=prices_item1, reward_item2=np.zeros(4))
            ts_learner_item1 = TS_Learner_item1(n_arms=n_arms, daily_customers=daily_customers, prices=prices_item1, reward_item2=np.zeros(4))

            ts_learner_item1_NS = SWTS_Learner(n_arms=n_arms, daily_customers=daily_customers, prices=prices_item1, reward_item2=np.zeros(4), window_size=window_size)


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
        # Number of arms for pricing item 1
        n_arms = 9

        # Candidate prices for item 1 and item 2 (one per arm)
        prices_item1 = self.data.prices_item1
        prices_item2 = self.data.prices_item2

        # Conversion rates for item 1 (one per arm)
        conversion_rates_item1 = self.data.conversion_rates_item1

        conversion_rates_item1_NS = np.array([conversion_rates_item1 + 0.2,
                                              conversion_rates_item1 - 0.2])
        conversion_rates_item1_NS = np.clip(conversion_rates_item1_NS, 0, 1)
        # dim (2,4,9): 2 phases - 4 customers classes - 9 prices

        # Conversion rates for item 2 (one 4x4 matrix per arm)
        conversion_rates_item2 = np.array([self.data.conversion_rates_item21 + 0.1,
                                           self.data.conversion_rates_item21,
                                           self.data.conversion_rates_item21 - 0.1])

        conversion_rates_item2_NS = np.array([conversion_rates_item2 + 0.2,
                                              conversion_rates_item2 - 0.2])
        conversion_rates_item2_NS = np.clip(conversion_rates_item2_NS, 0, 1)
        # dim (2,3,4,4): 2 phases - 3 full prices (40€-50€-60€) - 4 promos - 4 customers classes

        # Number of daily customers per class
        daily_customers = self.data.daily_customers

        # Parameters for the experiments
        n_experiments = 50
        time_horizon = 3000

        # Parameters for Non-Stationary Environment
        n_phases = 2
        phases_len = int(time_horizon / n_phases)

        # Promo assigment weights (2 phases - 3 item_2 prices - 4 promos - 4 customer classes)
        daily_promos = (promo_fractions * sum(daily_customers)).astype(int)
        weights = np.zeros((n_phases, 3, 4, 4))
        for i in range(n_phases):
            for j in range(3):
                weights[i, j, :, :] = normalize(lp.matching_lp(prices_item2[j], self.discounts,
                                                               conversion_rates_item2_NS[i, j], daily_promos,
                                                               daily_customers)[1], 'l1', axis=0)

        # Computing the objective matrix (one element per arm, composed by 2 phases and 3 item_2 prices)
        objective = np.zeros((n_arms, n_phases, 3))
        for i in range(n_arms):
            for j in range(n_phases):
                for k in range(3):
                    objective[i, j, k] = sum(prices_item1[i] * daily_customers * conversion_rates_item1_NS[j, :, i] +
                                             prices_item2[k] * daily_customers * conversion_rates_item1_NS[j, :, i] *
                                             (np.dot(1-self.discounts, conversion_rates_item2_NS[j, k, :, :] *
                                                     weights[j, k, :, :])))

        # Storing the optimal objective value to compute the regret later
        opt = np.amax(np.amax(objective, axis=2), axis=0)

        # Launching the experiments, using change detection
        # Four UCB1 (one for each customer class) are used to learn the conversion rates of item 2
        # One UCB1 is used to learn the best promo-customer class matching
        # One UCB1 is used to learn the price for item 1
        M = 1000
        eps = 0.1
        h = np.log(time_horizon) * 2

        ucb1_rewards_per_experiment_item1 = []

        for e in range(n_experiments):
            print("Experiment ", e+1, "/", n_experiments)

            # Environment and learner for the daily number of customers
            env_daily_customers = Daily_Customers(mean=daily_customers, sd=25)
            learner_daily_customers = Learner_Customers()
            daily_customers_empirical_means = np.zeros(4)

            # Environments and learners for the conversion rates of item 2
            envs_item2_NS = np.array([
                Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 0].reshape(n_phases, 12), horizon=time_horizon),
                Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 1].reshape(n_phases, 12), horizon=time_horizon),
                Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 2].reshape(n_phases, 12), horizon=time_horizon),
                Non_Stationary_Environment_First(n_arms=12, probabilities=conversion_rates_item2_NS[:, :, :, 3].reshape(n_phases, 12), horizon=time_horizon)])

            ucb1_learners_item2 = np.array([CUMSUM_UCB1_item2(n_arms=12, M=M, eps=eps, h=h),
                                            CUMSUM_UCB1_item2(n_arms=12, M=M, eps=eps, h=h),
                                            CUMSUM_UCB1_item2(n_arms=12, M=M, eps=eps, h=h),
                                            CUMSUM_UCB1_item2(n_arms=12, M=M, eps=eps, h=h)])

            # Environment and learner for the promo-customer class matching
            probabilities = np.zeros((3, 4))
            ucb1_learner_matching = CUMSUM_UCB_Matching(probabilities.size, *probabilities.shape, price=0, daily_customers=daily_customers, discounts=1-self.discounts, p_frac=promo_fractions, M=M, eps=eps, h=h)
            env_matching = Non_Stationary_Environment_First(probabilities.size, probabilities, time_horizon)

            # Environment and learner for the price of item 1
            env_item1_NS = Non_Stationary_Environment_Third(n_arms, conversion_rates_item1_NS, time_horizon)
            ucb1_learner_item1 = CUMSUM_UCB1_item1(n_arms, daily_customers, prices_item1, reward_item2=np.zeros(4), M=M, eps=0.05, h=20, alpha=0.01)

            for t in range(time_horizon):
                # Learning the number of customers
                daily_customers_sample = env_daily_customers.sample()
                daily_customers_empirical_means = learner_daily_customers.update_daily_customers(
                    empirical_means=daily_customers_empirical_means, sample=daily_customers_sample)

                # Learning the conversion rates for item 2
                for learner, env in zip(ucb1_learners_item2, envs_item2_NS):
                    pulled_arm = learner.pull_arm()
                    reward = env.round(pulled_arm)
                    learner.update(pulled_arm, reward)

                # Taking the best price for item 2 for each class
                # Each "empirical_means" is reshaped into a 3x4 matrix (3 prices for item 2, 4 conversion rates)
                indices_best_price_item2 = np.zeros(4)
                for i in range(4):
                    indices_best_price_item2[i] = np.argmax(
                        np.dot(ucb1_learners_item2[i].empirical_means.reshape(3, 4), (1-self.discounts)) * prices_item2,
                        axis=0)

                # Computing the most common price selected, since we want the same price for all the customer classes
                best_price_item2 = stats.mode(indices_best_price_item2)[0][0].astype(int)

                # Constructing the usual 4x4 matrix of conversion rates for item 2
                conversion_rates_item2_em = np.zeros((4, 4))
                for i in range(4):
                    conversion_rates_item2_em[:, i] = ucb1_learners_item2[i].empirical_means.reshape(3, 4)[best_price_item2]

                # Updating the probabilities vector in the environment used by UCB_matching
                # as well as the best price for item 2 that has been selected, and the number of customers learned
                env_matching.probabilities = conversion_rates_item2_em[1:, :]
                ucb1_learner_matching.price = prices_item2[best_price_item2]
                ucb1_learner_matching.daily_customers = daily_customers_empirical_means

                # Learning the best promo-customer class matching
                pulled_arms = ucb1_learner_matching.pull_arm()
                rewards = env_matching.round(pulled_arms)
                ucb1_learner_matching.update(pulled_arms, rewards)

                # Given the resulting pulled arms [[x, y, z], [a, b, c]], we reconstruct the usual 4x4 matrix
                # In the cells selected by the matching above, we give the promo (as a fraction of the customers)
                # Notice that the "+1" on the rows is required since the matching did not consider promo P0
                weights = np.zeros((4, 4))
                for i in range(0, 3):
                    for j in range(0, 4):
                        weights[pulled_arms[0][i]+1, pulled_arms[1][i]] = promo_fractions[i+1] * sum(daily_customers_empirical_means)

                # Otherwise, as always, we give promo P0 to the remaining customers
                for j in range(0, 4):
                    weights[0, j] = daily_customers_empirical_means[j] - sum(weights[:, j])

                # Normalizing the weights matrix to have proper values between 0 and 1
                weights = normalize(weights, 'l1', axis=0)

                # Computing the reward obtained when buying item 2 (one element per customer class)
                reward_item2 = np.zeros(4)
                for i in range(4):
                    reward_item2[i] = sum(prices_item2[best_price_item2] * (1-self.discounts) *
                                          conversion_rates_item2_em[:, i] * weights[:, i])

                # Updating the number of customers and the reward given by item 2 in the learner for the price of item 1
                ucb1_learner_item1.daily_customers = daily_customers_empirical_means
                ucb1_learner_item1.reward_item2 = reward_item2

                # Learning the best price for item 1
                pulled_arm = ucb1_learner_item1.pull_arm()
                reward = env_item1_NS.round(pulled_arm)
                ucb1_learner_item1.update(pulled_arm, reward)

            ucb1_rewards_per_experiment_item1.append(ucb1_learner_item1.collected_rewards)

        # Final computations to compute and plot the reward and the regret of UCB for price item 1
        ucb1_instantaneous_regret = np.zeros(time_horizon)
        optimum_per_round = np.zeros(time_horizon)

        for i in range(0, n_phases):
            t_index = range(i * phases_len, (i+1) * phases_len)
            optimum_per_round[t_index] = opt[i]
            ucb1_instantaneous_regret[t_index] = opt[i] - np.mean(ucb1_rewards_per_experiment_item1, axis=0)[t_index]

        # Plotting the regret and the reward
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Reward")
        plt.plot(np.mean(ucb1_rewards_per_experiment_item1, axis=0), 'r')
        plt.plot(optimum_per_round, '--k')
        plt.legend(["CUMSUM-UCB1", "Optimum"], title="STEP 8")
        plt.show()

        plt.figure(1)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(ucb1_instantaneous_regret), 'r')
        plt.legend(["CUMSUM-UCB1", "Optimum"], title="STEP 8")
        plt.show()
