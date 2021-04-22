import numpy as np
from LP_optimization import *
np.random.seed(1234)


class Environment:
    def __init__(self, n_arms, objective):
        self.n_arms = n_arms
        self.objective = objective

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.objective[pulled_arm])
        return reward
