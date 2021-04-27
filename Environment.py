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
    def __init__(self, n_arms, probabilities, candidates):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.candidates = candidates

    def round(self, pulled_arm):
        reward = (np.random.binomial(1, self.probabilities[pulled_arm]) * self.candidates[pulled_arm]) / max(self.candidates)
        return reward
