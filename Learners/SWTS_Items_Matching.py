import numpy as np
import itertools
import matching_lp as lp
np.random.seed(1234)


class SWTS_Items_Matching():
    def __init__(self, window_size, margins_item1, margins_item2, daily_customers, discounts, promo_fractions):
        self.t = 0
        self.window_size = window_size
        self.collected_rewards = np.array([])
        self.rewards_per_arm_item1 = [[[] for _ in range(4)] for _ in range(len(margins_item1))]
        self.rewards_per_arm_item2 = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(len(margins_item2))]

        self.margins = list(itertools.product(margins_item1, margins_item2))
        self.empirical_means_item1 = np.zeros((len(margins_item1), 4))
        self.beta_parameters_item1 = np.ones((len(margins_item1), 4, 2))
        self.empirical_means_item2 = np.zeros((len(margins_item2), 4, 4))
        self.beta_parameters_item2 = np.ones((len(margins_item2), 4, 4, 2))

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
            revenue = np.zeros((len(self.margins_item1), len(self.margins_item2)))
            for margin1 in range(len(self.margins_item1)):
                for margin2 in range(len(self.margins_item2)):
                    daily_promos = (self.promo_fractions * sum(self.daily_customers * (self.empirical_means_item1[margin1]))).astype(int)
                    revenue[margin1][margin2] = self.margins_item1[margin1] * (self.daily_customers * (self.empirical_means_item1[margin1])).sum() +\
                                                   lp.matching_lp(self.margins_item2[margin2], self.discounts, (self.empirical_means_item2[margin2]),
                                                                  daily_promos, (self.daily_customers * (self.empirical_means_item1[margin1])).astype(int))[0]

            arm_flat = np.argmax(np.random.random(revenue.shape) * (revenue == np.amax(revenue, None, keepdims=True)), None)
            arm = np.unravel_index(arm_flat, revenue.shape)

            selected_margin_item2 = self.margins_item2[arm[1]]
            daily_promos = (self.promo_fractions * sum(self.daily_customers * (self.empirical_means_item1[arm[0]]))).astype(int)
            _, matching = lp.matching_lp(selected_margin_item2, self.discounts, (self.empirical_means_item2[arm[1]]),
                                         daily_promos, (self.daily_customers * (self.empirical_means_item1[arm[0]])).astype(int))

        return arm, matching

    def update(self, pulled_arm, reward):
        self.t += 1

        # Item 1
        for class_type in range(4):
            self.rewards_per_arm_item1[pulled_arm[0]][class_type].append(reward[0][class_type])
            A = self.rewards_per_arm_item1[pulled_arm[0]][class_type]
            B = self.rewards_per_arm_item1[pulled_arm[0]][class_type][-self.window_size:]
            cum_rew_item1 = np.sum(self.rewards_per_arm_item1[pulled_arm[0]][class_type][-self.window_size:])
            n_rounds_arm_item1 = len(self.rewards_per_arm_item1[pulled_arm[0]][class_type][-self.window_size:])
            self.beta_parameters_item1[pulled_arm[0], class_type, 0] = cum_rew_item1 + 1.0
            self.beta_parameters_item1[pulled_arm[0], class_type, 1] = n_rounds_arm_item1 - cum_rew_item1 + 1.0
            sample = np.random.beta(self.beta_parameters_item1[pulled_arm[0], class_type, 0],
                                    self.beta_parameters_item1[pulled_arm[0], class_type, 1])
            self.empirical_means_item1[pulled_arm[0], class_type] = (self.empirical_means_item1[pulled_arm[0], class_type] *
                                                                                 (self.t - 1) + sample) / self.t

        # Item 2
        for promo_type in range(4):
            for class_type in range(4):
                self.rewards_per_arm_item2[pulled_arm[1]][promo_type][class_type].append(reward[1][promo_type][class_type])
                A = self.rewards_per_arm_item2[pulled_arm[1]][promo_type][class_type]
                B = self.rewards_per_arm_item2[pulled_arm[1]][promo_type][class_type][-self.window_size:]
                cum_rew_item2 = np.sum(self.rewards_per_arm_item2[pulled_arm[1]][promo_type][class_type][-self.window_size:])
                n_rounds_arm_item2 = len(self.rewards_per_arm_item2[pulled_arm[1]][promo_type][class_type][-self.window_size:])
                self.beta_parameters_item2[pulled_arm[1], promo_type, class_type, 0] = cum_rew_item2 + 1.0
                self.beta_parameters_item2[pulled_arm[1], promo_type, class_type, 1] = n_rounds_arm_item2 - cum_rew_item2 + 1.0
                sample = np.random.beta(self.beta_parameters_item2[pulled_arm[1], promo_type, class_type, 0],
                                        self.beta_parameters_item2[pulled_arm[1], promo_type, class_type, 1])
                self.empirical_means_item2[pulled_arm[1], promo_type, class_type] = (self.empirical_means_item2[pulled_arm[1], promo_type, class_type] *
                                                                                     (self.t - 1) + sample) / self.t

