from UCB import UCB
import numpy as np
from scipy.optimize import linear_sum_assignment


class UCB_Matching(UCB):
    def __init__(self, n_arms, n_rows, n_cols, price, discounts, p_frac):
        super().__init__(n_arms)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.price = price
        self.discounts = discounts
        self.p_frac = p_frac

        self.upper_conf = np.zeros((n_rows, n_cols))
        assert n_arms == n_cols * n_rows

    def pull_arm(self):
        #self.upper_conf = self.price * (self.discounts * self.p_frac).trasnpose() * (self.empirical_means + self.confidence).reshape(self.n_rows, self.n_cols)
        self.upper_conf[0,:] = self.price * self.discounts[0] * self.p_frac[0] * (self.empirical_means + self.confidence).reshape(self.n_rows, self.n_cols)[0,:]
        self.upper_conf[1, :] = self.price * self.discounts[1] * self.p_frac[1] * (self.empirical_means + self.confidence).reshape(self.n_rows, self.n_cols)[1,:]
        self.upper_conf[2, :] = self.price * self.discounts[2] * self.p_frac[2] * (self.empirical_means + self.confidence).reshape(self.n_rows, self.n_cols)[2,:]

        self.upper_conf[np.isinf(self.upper_conf)] = 1e3
        row_ind, col_ind = linear_sum_assignment(-self.upper_conf) #.reshape(self.n_rows, self.n_cols))
        return row_ind, col_ind

    def update(self, pulled_arms, rewards):
        self.t += 1
        pulled_arm_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        for pulled_arm, reward in zip(pulled_arm_flat, rewards):
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(self.t-1) + reward)/self.t

    def set_price(self, price):
        self.price = price
