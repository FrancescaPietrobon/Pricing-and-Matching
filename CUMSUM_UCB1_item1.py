from UCB1_item1 import UCB1_item1
from Cumsum import CUMSUM
import numpy as np
np.random.seed(1234)


class CUMSUM_UCB1_item1(UCB1_item1):
    def __init__(self, n_arms, daily_customers, margins, reward_item2, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(n_arms, daily_customers, margins, reward_item2)
        self.change_detection = [CUMSUM(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arms = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha

    def pull_arm(self):
        if self.t < self.n_arms:
            arm = self.t
        else:
            if np.random.binomial(1, 1 - self.alpha):
                upper_bound = (self.margins_item1 * np.dot(self.daily_customers, (self.empirical_means + self.confidence))) + np.dot(self.daily_customers * self.reward_item2, (self.empirical_means + self.confidence))
                arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
            else:
                arm = np.random.randint(0, self.n_arms)
        return arm

    def update(self, pulled_arm, reward):
        self.t += 1
        if self.change_detection[pulled_arm].update(reward):
            self.detections[pulled_arm].append(self.t)
            self.valid_rewards_per_arms[pulled_arm] = []
            self.change_detection[pulled_arm].reset()
        self.update_observations(pulled_arm, reward)
        self.empirical_means[:, pulled_arm] = (self.empirical_means[:, pulled_arm] * (self.t - 1) + reward) / self.t
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arms])
        for a in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arms[a])
            self.confidence[a] = (2 * np.log(total_valid_samples) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arms[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, np.sum((self.margins_item1[pulled_arm] + self.reward_item2) * self.daily_customers * reward))
