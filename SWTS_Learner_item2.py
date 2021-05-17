from Learner import *
from TS_Learner_item2 import *


class SWTS_Learner_item2(Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.empirical_means = np.ones(n_arms)
        self.window_size = window_size

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        self.empirical_means = self.beta_parameters[:, 0] / (self.beta_parameters[:, 0] + self.beta_parameters[:, 1])
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        cum_rew = np.sum(self.rewards_per_arm[pulled_arm][-self.window_size:])
        n_rounds_arm = len(self.rewards_per_arm[pulled_arm][-self.window_size:])

        self.beta_parameters[pulled_arm, 0] = cum_rew + 1.0
        self.beta_parameters[pulled_arm, 1] = n_rounds_arm - cum_rew + 1.0

    def get_empirical_means(self):
        return self.empirical_means
