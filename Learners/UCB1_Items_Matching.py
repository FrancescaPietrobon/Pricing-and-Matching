import numpy as np
import itertools
import matching_lp as lp
np.random.seed(1234)


class UCB1_Items_Matching():
    def __init__(self, margins_item1, margins_item2, daily_customers, discounts, promo_fractions):
        self.t = 0
        self.collected_rewards = np.array([])
        self.rewards_per_arm_item1 = [[[] for _ in range(4)] for _ in range(len(margins_item1))]
        self.rewards_per_arm_item2 = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(len(margins_item2))]

        self.margins = list(itertools.product(margins_item1, margins_item2))
        self.empirical_means_item1 = np.zeros((len(margins_item1), 4))
        self.confidence_item1 = np.full((len(margins_item1), 4), np.inf)
        self.empirical_means_item2 = np.zeros((len(margins_item2), 4, 4))
        self.confidence_item2 = np.full((len(margins_item2), 4, 4), np.inf)

        self.margins_item1 = margins_item1
        self.margins_item2 = margins_item2
        self.daily_customers = daily_customers
        self.discounts = discounts
        self.promo_fractions = promo_fractions

    def pull_arm(self):
        if self.t < len(self.margins):
            arm = np.unravel_index(self.t, (len(self.margins_item1), len(self.margins_item2)))
            matching = np.random.rand(4, 4)
        else:
            upper_conf = np.zeros((len(self.margins_item1), len(self.margins_item2)))
            matching = np.zeros((len(self.margins_item1), len(self.margins_item2), 4, 4))

            for margin1 in range(len(self.margins_item1)):
                for margin2 in range(len(self.margins_item2)):
                    daily_promos = (self.promo_fractions * sum(self.daily_customers * (self.empirical_means_item1[margin1] + self.confidence_item1[margin1]))).astype(int)
                    reward_item2, matching[margin1][margin2] = lp.matching_lp(self.margins_item2[margin2], self.discounts, (self.empirical_means_item2[margin2] + self.confidence_item2[margin2]),
                                                                              daily_promos, (self.daily_customers * (self.empirical_means_item1[margin1] + self.confidence_item1[margin1])).astype(int))
                    upper_conf[margin1][margin2] = self.margins_item1[margin1] * (self.daily_customers * (self.empirical_means_item1[margin1] + self.confidence_item1[margin1])).sum() + reward_item2

            arm_flat = np.argmax(np.random.random(upper_conf.shape) * (upper_conf == np.amax(upper_conf, None, keepdims=True)), None)
            arm = np.unravel_index(arm_flat, upper_conf.shape)
            matching = matching[arm[0]][arm[1]]

        return arm, matching

    def update(self, pulled_arm, reward):
        self.t += 1

        # Confidence item 1
        for margin1 in range(len(self.margins_item1)):
            for class_type in range(4):
                number_pulled = max(1, len(self.rewards_per_arm_item1[margin1][class_type]))
                self.confidence_item1[margin1][class_type] = (2*np.log(self.t) / number_pulled)**0.5

        # Confidence item 2
        for margin2 in range(len(self.margins_item2)):
            for promo_type in range(4):
                for class_type in range(4):
                    number_pulled = max(1, len(self.rewards_per_arm_item2[margin2][promo_type][class_type]))
                    self.confidence_item2[margin2][promo_type][class_type] = (2*np.log(self.t) / number_pulled)**0.5

        # Empirical means item 1
        self.empirical_means_item1[pulled_arm[0]] = (self.empirical_means_item1[pulled_arm[0]] * (self.t-1) + reward[0]) / self.t
        for class_type in range(4):
            self.rewards_per_arm_item1[pulled_arm[0]][class_type].append(reward[0][class_type])

        # Empirical means item 2
        self.empirical_means_item2[pulled_arm[1]] = (self.empirical_means_item2[pulled_arm[1]] * (self.t-1) + reward[1]) / self.t
        for promo_type in range(4):
            for class_type in range(4):
                self.rewards_per_arm_item2[pulled_arm[1]][promo_type][class_type].append(reward[1][promo_type][class_type])
