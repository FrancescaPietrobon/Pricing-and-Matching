import numpy as np
import math
from sklearn.preprocessing import normalize
import itertools
np.random.seed(1234)


class Environment_Step3:
    def __init__(self, n_arms, conversion_rates_item1, daily_customers):
        self.n_arms = n_arms
        self.conversion_rates_item1 = conversion_rates_item1
        self.daily_customers = daily_customers

    def round(self, pulled_arm):
        reward = np.zeros(4)
        offer = np.zeros(4)
        for i in range(sum(self.daily_customers)):
            group = np.random.choice(4, p=[self.daily_customers[0] / sum(self.daily_customers),
                                           self.daily_customers[1] / sum(self.daily_customers),
                                           self.daily_customers[2] / sum(self.daily_customers),
                                           self.daily_customers[3] / sum(self.daily_customers)])
            offer[group] += 1
            reward[group] = reward[group] + np.random.binomial(1, self.conversion_rates_item1[group, pulled_arm])

        result = np.zeros(4)
        for i in range(4):
            result[i] = reward[i] / offer[i] if offer[i] > 0 else 0

        return result


class Environment_Step4:
    def __init__(self, n_arms, conversion_rates_item1, conversion_rates_item2, weights, daily_customers):
        self.n_arms = n_arms
        self.conversion_rates_item1 = conversion_rates_item1
        self.conversion_rates_item2 = conversion_rates_item2
        self.weights = weights
        self.daily_customers = daily_customers

    def round(self, pulled_arm):
        reward1 = np.zeros(4)
        reward2 = np.zeros((4, 4))
        offer1 = np.zeros(4)
        offer2 = np.zeros((4, 4))
        for i in range(sum(self.daily_customers)):
            group = np.random.choice(4, p=[self.daily_customers[0] / sum(self.daily_customers),
                                           self.daily_customers[1] / sum(self.daily_customers),
                                           self.daily_customers[2] / sum(self.daily_customers),
                                           self.daily_customers[3] / sum(self.daily_customers)])
            bin_item1 = np.random.binomial(1, self.conversion_rates_item1[group, pulled_arm])
            offer1[group] += 1
            reward1[group] = reward1[group] + bin_item1
            if bin_item1 == 1:
                promo = np.random.choice(4, p=self.weights[:, group])
                offer2[promo, group] += 1
                reward2[promo, group] = reward2[promo, group] + np.random.binomial(1, self.conversion_rates_item2[promo, group])

        result1 = np.zeros(4)
        for i in range(4):
            result1[i] = reward1[i] / offer1[i] if offer1[i] > 0 else 0

        result2 = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                result2[i, j] = reward2[i, j] / offer2[i, j] if offer2[i, j] > 0 else 0

        return result1, result2


class Environment_Step5:
    def __init__(self, n_arms, conversion_rates_item2, daily_customers, promo_fractions):
        self.n_arms = n_arms
        self.conversion_rates_item2 = conversion_rates_item2
        self.daily_customers = daily_customers
        self.promo_fractions = promo_fractions

    def round(self, pulled_arms):
        # Given the resulting pulled arms [[x, y, z], [a, b, c]], we reconstruct the usual 4x4 matrix
        # In the cells selected by the matching above, we give the promo (as a fraction of the customers)
        # Notice that the "+1" on the rows is required since the matching did not consider promo P0
        weights = np.zeros((4, 4))
        for i in range(0, 3):
            if (self.promo_fractions[i+1] * sum(self.daily_customers)) <= self.daily_customers[pulled_arms[1][i]]:
                weights[pulled_arms[0][i]+1, pulled_arms[1][i]] = self.promo_fractions[i+1] * sum(self.daily_customers)
            else:
                weights[pulled_arms[0][i]+1, pulled_arms[1][i]] = self.daily_customers[pulled_arms[1][i]]

        # Otherwise, as always, we give promo P0 to the remaining customers
        for j in range(0, 4):
            weights[0, j] = self.daily_customers[j] - sum(weights[:, j])

        # Normalizing the weights matrix to have proper values between 0 and 1
        weights = normalize(weights, 'l1', axis=0)

        # Simulating the arrival of customers (that buy item 2)
        reward2 = np.zeros((4, 4))
        offer = np.zeros((4, 4))
        for i in range(sum(self.daily_customers)):
            group = np.random.choice(4, p=[self.daily_customers[0] / sum(self.daily_customers),
                                           self.daily_customers[1] / sum(self.daily_customers),
                                           self.daily_customers[2] / sum(self.daily_customers),
                                           self.daily_customers[3] / sum(self.daily_customers)])
            promo = np.random.choice(4, p=weights[:, group])
            offer[promo, group] += 1
            reward2[promo, group] = reward2[promo, group] + np.random.binomial(1, self.conversion_rates_item2[promo, group])

        result = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                result[i, j] = reward2[i, j] / offer[i, j] if offer[i, j] > 0 else 0

        result = result[1:]
        result = np.sum(result, axis=1)

        return result


class Environment_Step6:
    def __init__(self, n_arms, conversion_rates_item1, conversion_rates_item2,
                 daily_customers, promo_fractions, margins_item1, margins_item2):
        self.n_arms = n_arms
        self.conversion_rates_item1 = conversion_rates_item1
        self.conversion_rates_item2 = conversion_rates_item2
        self.daily_customers = daily_customers
        self.promo_fractions = promo_fractions
        self.margins_item1 = margins_item1
        self.margins_item2 = margins_item2

    def round(self, pulled_arm):

        idx_price_item1 = list(self.margins_item1).index(pulled_arm[0][0])
        idx_price_item2 = list(self.margins_item2).index(pulled_arm[0][1])

        weights = np.zeros((4, 4))
        for i in range(0, 3):
            if (self.promo_fractions[i + 1] * sum(self.daily_customers)) <= self.daily_customers[pulled_arm[2][i]]:
                weights[pulled_arm[1][i] + 1, pulled_arm[2][i]] = self.promo_fractions[i + 1] * sum(
                    self.daily_customers)
            else:
                weights[pulled_arm[1][i] + 1, pulled_arm[2][i]] = self.daily_customers[pulled_arm[2][i]]

        # Otherwise, as always, we give promo P0 to the remaining customers
        for j in range(0, 4):
            weights[0, j] = self.daily_customers[j] - sum(weights[:, j])

        # Normalizing the weights matrix to have proper values between 0 and 1
        weights = normalize(weights, 'l1', axis=0)

        # Simulating the arrival of customers
        reward1 = np.zeros(4)
        reward2 = np.zeros((4, 4))
        offer1 = np.zeros(4)
        offer2 = np.zeros((4, 4))
        for i in range(sum(self.daily_customers)):
            group = np.random.choice(4, p=[self.daily_customers[0] / sum(self.daily_customers),
                                           self.daily_customers[1] / sum(self.daily_customers),
                                           self.daily_customers[2] / sum(self.daily_customers),
                                           self.daily_customers[3] / sum(self.daily_customers)])
            bin_item1 = np.random.binomial(1, self.conversion_rates_item1[group, idx_price_item1])
            offer1[group] += 1
            reward1[group] = reward1[group] + bin_item1
            if bin_item1 == 1:
                promo = np.random.choice(4, p=weights[:, group])
                offer2[promo, group] += 1
                reward2[promo, group] = reward2[promo, group] + np.random.binomial(1, self.conversion_rates_item2[idx_price_item2,
                    promo, group])

        result1 = np.zeros(4)
        for i in range(4):
            result1[i] = reward1[i] / offer1[i] if offer1[i] > 0 else 0

        result2 = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                result2[i, j] = reward2[i, j] / offer2[i, j] if offer2[i, j] > 0 else 0

        result3 = result2[1:]

        return [result1, result2, result3]

########################################################################################################################
class Daily_Customers:
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def sample(self):
        return np.clip(np.random.normal(self.mean, self.sd), 0, 500).astype(int)


class Non_Stationary_Environment_First:
    def __init__(self, n_arms, probabilities, horizon):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.t = 0
        n_phases = np.shape(self.probabilities)[0]
        self.phases_size = horizon/n_phases

    def round(self, pulled_arm):
        current_phase = int(self.t / self.phases_size)
        p = self.probabilities[current_phase, pulled_arm]
        reward = np.random.binomial(1, p)
        self.t += 1
        return reward


class Non_Stationary_Environment_Third:
    def __init__(self, n_arms, probabilities, horizon):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.time = 0
        n_phases = np.shape(self.probabilities)[0]
        self.phases_size = int(horizon/n_phases)

    def round(self, pulled_arm):
        current_phase = math.floor(self.time / self.phases_size)
        p = self.probabilities[current_phase, :, pulled_arm]
        reward = np.random.binomial(1, p)
        self.time += 1
        return reward
