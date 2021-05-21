from UCB_matching import UCB_Matching
import numpy as np
from scipy.optimize import linear_sum_assignment
from Cumsum import CUMSUM


class CUMSUM_UCB_Matching(UCB_Matching):
    def __init__(self, n_arms, n_rows, n_cols, price, daily_customers, discounts, p_frac, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(n_arms, n_rows, n_cols, price, daily_customers, discounts, p_frac)
        self.change_detection = [CUMSUM(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arms = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha

    def pull_arm(self):
        if np.random.binomial(1, 1-self.alpha):
            for i in range(self.n_rows):
                self.upper_conf[i, :] = self.price * self.discounts[i + 1] * self.p_frac[i + 1] * self.daily_customers * \
                                        (self.empirical_means + self.confidence).reshape(self.n_rows, self.n_cols)[i, :]
            self.upper_conf[np.isinf(self.upper_conf)] = 1e3
            row_ind, col_ind = linear_sum_assignment(-self.upper_conf)
            return row_ind, col_ind
        else:
            costs_random = np.random.randint(0, 10, size=(self.n_rows, self.n_cols))
            return linear_sum_assignment(costs_random)

    def update(self, pulled_arms, rewards):
        self.t += 1
        pulled_arm_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        for pulled_arm, reward in zip(pulled_arm_flat, rewards):
            if self.change_detection[pulled_arm].update(reward):
                self.detections[pulled_arm].append(self.t)
                self.valid_rewards_per_arms[pulled_arm] = []
                self.change_detection[pulled_arm].reset()
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arms[pulled_arm])
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arms])
        for a in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arms[a])
            self.confidence[a] = (2 * np.log(total_valid_samples) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arms[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
