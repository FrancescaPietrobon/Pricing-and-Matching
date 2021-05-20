from Learner import *
from TS_Learner_item2 import *


class SWTS_Learner_item2(Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.empirical_means = np.zeros(n_arms)
        self.window_size = window_size
        self.pull_arms = np.array([])

    def pull_arm(self):
        if self.t < self.n_arms:
            idx = self.t
        else:
            sample = np.zeros(self.n_arms)
            sample = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
            idx = np.argmax(sample)
            # self.empirical_means = (self.empirical_means * (self.t-1) + sample) / self.t
            self.empirical_means = self.beta_parameters[:, 0] / (self.beta_parameters[:, 0] + self.beta_parameters[:, 1])
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        cum_rew = np.sum(self.rewards_per_arm[pulled_arm][-self.window_size:])
        n_rounds_arm = len(self.rewards_per_arm[pulled_arm][-self.window_size:])

        self.beta_parameters[pulled_arm, 0] = cum_rew + 1.0
        self.beta_parameters[pulled_arm, 1] = n_rounds_arm - cum_rew + 1.0

    '''
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.pull_arms = np.append(self.pull_arms, pulled_arm)
        for arm in range(self.n_arms):
            n_samples = np.sum(self.pull_arms[-self.window_size:] == arm)
            cum_rew = np.sum(self.rewards_per_arm[arm][-n_samples:], axis=0) if n_samples > 0 else 0
            self.beta_parameters[arm, 0] = cum_rew + 1.0
            self.beta_parameters[arm, 1] = n_samples - cum_rew + 1.0
    '''

    def get_empirical_means(self):
        return self.empirical_means
