import numpy as np
import itertools
from scipy.optimize import linear_sum_assignment
np.random.seed(1234)

class UCB1_items_matching():
    def __init__(self, n_arms,  n_rows, n_cols, daily_customers, margins_item1, margins_item2, discounts, p_frac):
        self.n_arms = n_arms
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.t = 0
        self.margins = list(itertools.product(margins_item1, margins_item2))
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])
        self.empirical_means = np.zeros((len(self.margins), 2, 3, 4))
        self.confidence = np.zeros(n_arms)
        self.daily_customers = daily_customers
        self.margins_item1 = margins_item1
        self.margins_item2 = margins_item2
        self.discounts = discounts
        self.p_frac = p_frac

        self.upper_conf = np.zeros((len(self.margins), n_rows, n_cols))

    def pull_arm(self):
        gain = np.zeros(15)
        for cross in range(len(self.margins)):
            for i in range(self.n_rows):
                for class_type in range(self.n_cols):
                    arm = cross + i + class_type
                    #[15, 3, 4]
                    self.upper_conf[cross][i][class_type] = \
                                                    (self.margins[cross][0] * self.daily_customers[class_type] *
                                                     (self.empirical_means[cross][0][i][class_type] + self.confidence[arm])) \
                                                    + self.margins[cross][1] * self.discounts[i + 1] * self.p_frac[i + 1] \
                                                    * (self.empirical_means[cross][0][i][class_type] + self.confidence[arm]) \
                                                    * self.daily_customers[class_type]* (self.empirical_means[cross][1][i][class_type] +
                                                    self.confidence[arm])

        self.upper_conf[np.isinf(self.upper_conf)] = 1e3
        row_ind = np.zeros((len(self.margins), 3))
        col_ind = np.zeros((len(self.margins), 3))
        for i in range(len(self.margins)):
            row_ind[i], col_ind[i] = linear_sum_assignment(-self.upper_conf[i])

        #computing the gain for each couple of prices
        for i in range(len(self.margins)):
            for j in range(len(row_ind[i])):
                gain[i] = gain[i] + self.upper_conf[i, int(row_ind[i][j]), int(col_ind[i][j])]

        arm = np.random.choice(np.where(gain == gain.max())[0])
        return [arm, row_ind[arm].astype(int), col_ind[arm].astype(int)]

    def update(self, pulled_arm, reward):

        self.t += 1

        for a in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[a]))
            self.confidence[a] = (2*np.log(self.t) / number_pulled)**0.5

        pulled_arm_flat = np.ravel_multi_index([pulled_arm[1], pulled_arm[2]], (self.n_rows, self.n_cols))


        for pulled_arms, rew in zip(pulled_arm_flat, reward[2]):
           self.rewards_per_arm[pulled_arms].append(rew)

        for i in range(len(pulled_arm[1])):
            self.empirical_means[pulled_arm[0], 1, pulled_arm[1][i], pulled_arm[2][i]] = (self.empirical_means[pulled_arm[0], 1, pulled_arm[1][i], pulled_arm[2][i]] * (self.t - 1) + reward[2][pulled_arm[1][i]][pulled_arm[2][i]]) / self.t

        self.empirical_means[pulled_arm[0], 0, :, :] = (self.empirical_means[pulled_arm[0], 0, :, :] * (self.t - 1) + reward[0][:]) / self.t



