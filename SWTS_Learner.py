from TS_Learner_item1 import *
from TS_Learner_item2 import *


class SWTS_Learner(TS_Learner_item1):
    def __init__(self, n_arms, daily_customers, margins, reward_item2, window_size):
        super().__init__(n_arms, daily_customers, margins, reward_item2)
        self.window_size = window_size
        self.pull_arms = np.array([])

    '''
    def update(self, pulled_arm, reward):
        self.t += 1
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, np.sum((self.prices[pulled_arm] + self.reward_item2) * self.daily_customers * reward))
        self.pull_arms = np.append(self.pull_arms, pulled_arm)
        for arm in range(self.n_arms):
            n_samples = np.sum(self.pull_arms[-self.window_size:] == arm)
            cum_rew = np.sum(self.rewards_per_arm[arm][-n_samples:], axis=0) if n_samples > 0 else 0
            self.beta_parameters[:, arm, 0] = cum_rew + 1.0
            self.beta_parameters[:, arm, 1] = n_samples - cum_rew + 1.0
    '''

    def update(self, pulled_arm, reward):
        self.t += 1
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, np.sum((self.margins[pulled_arm] + self.reward_item2) * self.daily_customers * reward))
        cum_rew = np.sum(self.rewards_per_arm[pulled_arm][-self.window_size:], axis=0)
        n_rounds_arm = len(self.rewards_per_arm[pulled_arm][-self.window_size:])

        self.beta_parameters[:, pulled_arm, 0] = cum_rew + 1.0
        self.beta_parameters[:, pulled_arm, 1] = n_rounds_arm - cum_rew + 1.0
