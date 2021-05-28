import numpy as np
import math
np.random.seed(1234)

# First case of Environment (standard one)
class Environment_First:
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward

    def set_probabilities(self, probabilities):
        self.probabilities = probabilities


# Second case of Environment, in which the reward is obtained considering also the candidates
class Environment_Second:
    def __init__(self, n_arms, conversion_rates_item1, customers):
        self.n_arms = n_arms
        self.conversion_rates_item1 = conversion_rates_item1
        self.customers = customers

    def round(self, pulled_arm):
        reward = np.zeros(4)
        for i in range(sum(self.customers)):
            group = np.random.choice(4, p=[self.customers[0] / sum(self.customers),
                                           self.customers[1] / sum(self.customers),
                                           self.customers[2] / sum(self.customers),
                                           self.customers[3] / sum(self.customers)])
            reward[group] = reward[group] + np.random.binomial(1, self.conversion_rates_item1[group, pulled_arm])
        reward = reward / sum(reward)
        return reward


class Environment_Third:
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[:, pulled_arm])
        return reward


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

