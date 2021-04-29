import numpy as np
np.random.seed(1234)

# First case of Environment (standard one)
class Environment_First:
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward


# Second case of Environment, in which the reward is obtained considering also the candidates
class Environment_Second:
    def __init__(self, n_arms, conversion_rates_item1, conversion_rates_item21, reward_item1, reward_item2):
        self.n_arms = n_arms
        self.conversion_rates_item1 = conversion_rates_item1
        self.conversion_rates_item21 = conversion_rates_item21
        self.reward_item1 = reward_item1
        self.reward_item2 = reward_item2

    def round(self, pulled_arm):
        bin_item1 = np.random.binomial(1, self.conversion_rates_item1[:, pulled_arm])
        reward1 = np.sum(bin_item1 * self.reward_item1[:, pulled_arm])
        reward2 = np.sum(np.random.binomial(1, self.conversion_rates_item21[:, :, pulled_arm]) * self.reward_item2)
        reward = (reward1 + reward2) / (np.sum(self.reward_item1) + np.sum(self.reward_item2))
        return reward
