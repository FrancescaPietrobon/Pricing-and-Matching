import numpy as np
import matching_lp as lp
from Learners.TS_Items_Matching import TS_Items_Matching

np.random.seed(1234)


class SWTS_Items_Matching(TS_Items_Matching):
    def __init__(self, window_size, margins_item1, margins_item2, daily_customers, discounts, promo_fractions):
        super().__init__(margins_item1, margins_item2, daily_customers, discounts, promo_fractions)

        self.window_size = window_size
        self.rewards_per_arm_item1 = [[[] for _ in range(4)] for _ in range(len(margins_item1))]
        self.rewards_per_arm_item2 = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(len(margins_item2))]

    def pull_arm(self):
        value = np.zeros((len(self.margins_item1), len(self.margins_item2)))
        matching = np.zeros((len(self.margins_item1), len(self.margins_item2), 4, 4))

        for margin1 in range(len(self.margins_item1)):
            beta_item1 = np.random.beta(self.beta_parameters_item1[margin1, :, 0], self.beta_parameters_item1[margin1, :, 1])
            for margin2 in range(len(self.margins_item2)):
                beta_item2 = np.random.beta(self.beta_parameters_item2[margin2, :, :, 0], self.beta_parameters_item2[margin2, :, :, 1])
                daily_promos = (self.promo_fractions * sum(self.daily_customers * beta_item1)).astype(int)
                reward_item2, matching[margin1][margin2] = lp.matching_lp(self.margins_item2[margin2], self.discounts, beta_item2, daily_promos, (self.daily_customers * beta_item1).astype(int))
                value[margin1][margin2] = self.margins_item1[margin1] * (self.daily_customers * beta_item1).sum() + reward_item2

        arm_flat = np.argmax(np.random.random(value.shape) * (value == np.amax(value, None, keepdims=True)), None)
        arm = np.unravel_index(arm_flat, value.shape)

        return arm, matching[arm[0]][arm[1]]

    def update(self, pulled_arm, reward):
        self.t += 1

        # Item 1
        for class_type in range(4):
            self.rewards_per_arm_item1[pulled_arm[0]][class_type].append(reward[0][class_type])
            cum_rew_item1 = np.sum(self.rewards_per_arm_item1[pulled_arm[0]][class_type][-self.window_size:])
            n_rounds_arm_item1 = len(self.rewards_per_arm_item1[pulled_arm[0]][class_type][-self.window_size:])
            self.beta_parameters_item1[pulled_arm[0], class_type, 0] = cum_rew_item1 + 1.0
            self.beta_parameters_item1[pulled_arm[0], class_type, 1] = n_rounds_arm_item1 - cum_rew_item1 + 1.0

        # Item 2
        for promo_type in range(4):
            for class_type in range(4):
                self.rewards_per_arm_item2[pulled_arm[1]][promo_type][class_type].append(reward[1][promo_type][class_type])
                cum_rew_item2 = np.sum(self.rewards_per_arm_item2[pulled_arm[1]][promo_type][class_type][-self.window_size:])
                n_rounds_arm_item2 = len(self.rewards_per_arm_item2[pulled_arm[1]][promo_type][class_type][-self.window_size:])
                self.beta_parameters_item2[pulled_arm[1], promo_type, class_type, 0] = cum_rew_item2 + 1.0
                self.beta_parameters_item2[pulled_arm[1], promo_type, class_type, 1] = n_rounds_arm_item2 - cum_rew_item2 + 1.0
