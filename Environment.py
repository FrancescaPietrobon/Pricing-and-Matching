import numpy as np
np.random.seed(1234)


class Environment:
    def __init__(self, n_arms, probabilities, objective):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.objective = objective

    def round(self, pulled_arm):
        reward = (np.random.binomial(1, self.probabilities[pulled_arm]) * self.objective[pulled_arm]) / max(self.objective)
        return reward
