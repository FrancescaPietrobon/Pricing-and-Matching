import numpy as np
from scipy.optimize import linear_sum_assignment


class UCB_Matching():
    def __init__(self, n_arms, n_rows, n_cols, price, daily_customers, discounts, p_frac):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf] * n_arms)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.price = price
        self.daily_customers = daily_customers
        self.discounts = discounts
        self.p_frac = p_frac

        self.upper_conf = np.zeros((n_rows, n_cols))
        assert n_arms == n_cols * n_rows

    def pull_arm(self):
        for i in range(self.n_rows):
            self.upper_conf[i, :] = self.price * self.discounts[i+1] * self.p_frac[i+1] * self.daily_customers *\
                                    (self.empirical_means + self.confidence).reshape(self.n_rows, self.n_cols)[i, :]
        self.upper_conf[np.isinf(self.upper_conf)] = 1e3
        row_ind, col_ind = linear_sum_assignment(-self.upper_conf)
        return row_ind, col_ind

    def update(self, pulled_arms, rewards):
        self.t += 1
        pulled_arm_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        for pulled_arm, reward in zip(pulled_arm_flat, rewards):
            self.rewards_per_arm[pulled_arm].append(reward)
            self.collected_rewards = np.append(self.collected_rewards, reward)
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(self.t-1) + reward)/self.t
