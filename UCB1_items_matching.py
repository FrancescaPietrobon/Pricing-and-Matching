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
        self.collected_rewards = np.array([])
        self.daily_customers = daily_customers
        self.margins_item1 = margins_item1
        self.margins_item2 = margins_item2
        self.discounts = discounts
        self.p_frac = p_frac

        self.rewards_per_arm_item1 = [[[] for _ in range(n_cols)] for _ in range(len(margins_item1))]
        self.rewards_per_arm_item2 = [[[[] for _ in range(n_cols)] for _ in range(n_rows)] for _ in range(len(margins_item2))]

        self.margins = list(itertools.product(margins_item1, margins_item2))
        self.empirical_means_item1 = np.zeros((len(margins_item1), n_cols))
        self.confidence_item1 = np.full((len(margins_item1), n_cols), np.inf)  # TODO put this also in the other bandits
        self.empirical_means_item2 = np.zeros((len(margins_item2), n_rows, n_cols))
        self.confidence_item2 = np.full((len(margins_item2), n_rows, n_cols), np.inf)
        self.upper_conf = np.zeros((len(self.margins_item1), len(self.margins_item2), n_rows, n_cols))

    def pull_arm(self):
        gain = np.zeros((len(self.margins_item1), len(self.margins_item2)))

        for price1 in range(len(self.margins_item1)):
            for price2 in range(len(self.margins_item2)):
                for promo_type in range(self.n_rows):
                    for class_type in range(self.n_cols):
                        self.upper_conf[price1][price2][promo_type][class_type] =\
                            (self.margins_item1[price1] * self.daily_customers[class_type] * (self.empirical_means_item1[price1][class_type] + self.confidence_item1[price1][class_type]) +
                             self.margins_item2[price2] * self.daily_customers[class_type] * (self.empirical_means_item1[price1][class_type] + self.confidence_item1[price1][class_type]) *
                             self.discounts[promo_type+1] * self.p_frac[promo_type+1] * (self.empirical_means_item2[price2][promo_type][class_type] + self.confidence_item2[price2][promo_type][class_type]))

        self.upper_conf[np.isinf(self.upper_conf)] = 1e3
        row_ind = np.zeros((len(self.margins_item1), len(self.margins_item2), 3))
        col_ind = np.zeros((len(self.margins_item1), len(self.margins_item2), 3))
        for price1 in range(len(self.margins_item1)):
            for price2 in range(len(self.margins_item2)):
                row_ind[price1][price2], col_ind[price1][price2] = linear_sum_assignment(-self.upper_conf[price1][price2])

        # Computing the gain for each couple of prices
        for price1 in range(len(self.margins_item1)):
            for price2 in range(len(self.margins_item2)):
                for i in range(3):
                    gain[price1][price2] = gain[price1][price2] + self.upper_conf[price1][price2][int(row_ind[price1][price2][i])][int(col_ind[price1][price2][i])]

        arm_idx1, arm_idx2 = np.unravel_index(np.argmax(gain), gain.shape)
        return [[arm_idx1, arm_idx2], row_ind[arm_idx1][arm_idx2].astype(int), col_ind[arm_idx1][arm_idx2].astype(int)]

    def update(self, pulled_arm, reward):
        self.t += 1

        for price1 in range(len(self.margins_item1)):
            for class_type in range(self.n_cols):
                number_pulled = max(1, len(self.rewards_per_arm_item1[price1][class_type]))
                self.confidence_item1[price1][class_type] = (2*np.log(self.t) / number_pulled)**0.5

        for price2 in range(len(self.margins_item2)):
            for promo_type in range(self.n_rows):
                for class_type in range(self.n_cols):
                    number_pulled = max(1, len(self.rewards_per_arm_item2[price2][promo_type][class_type]))
                    self.confidence_item2[price2][promo_type][class_type] = (2*np.log(self.t) / number_pulled)**0.5

        self.empirical_means_item1[pulled_arm[0][0]] = (self.empirical_means_item1[pulled_arm[0][0]] * (self.t-1) + reward[0]) / self.t
        self.rewards_per_arm_item1[pulled_arm[0][0]].append(reward[0])

        for i in range(3):
            self.rewards_per_arm_item2[pulled_arm[0][1]][pulled_arm[1][i]][pulled_arm[2][i]].append(reward[2][i])
            self.empirical_means_item2[pulled_arm[0][1]][pulled_arm[1][i]][pulled_arm[2][i]] = (self.empirical_means_item2[pulled_arm[0][1]][pulled_arm[1][i]][pulled_arm[2][i]] * (self.t - 1) + reward[2][i]) / self.t
