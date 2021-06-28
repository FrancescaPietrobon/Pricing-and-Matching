from Learners.UCB1_Items_Matching import *


class CD_UCB1_Items_Matching(UCB1_Items_Matching):
    def __init__(self, margins_item1, margins_item2, daily_customers, discounts, promo_fractions, M, eps, h, alpha):
        super().__init__(margins_item1, margins_item2, daily_customers, discounts, promo_fractions)

        self.change_detection_item1 = [[CUMSUM(M, eps, h) for _ in range(4)] for _ in range(len(margins_item1))]
        self.valid_rewards_per_arms_item1 = [[[] for _ in range(4)] for _ in range(len(margins_item1))]
        self.detections_item1 = [[[] for _ in range(4)] for _ in range(len(margins_item1))]

        self.change_detection_item2 = [[[CUMSUM(M, eps, h) for _ in range(4)] for _ in range(4)] for _ in range(len(margins_item2))]
        self.valid_rewards_per_arms_item2 = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(len(margins_item2))]
        self.detections_item2 = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(len(margins_item2))]

        self.alpha = alpha

    def pull_arm(self):
        if self.t < len(self.margins):
            arm = np.unravel_index(self.t, (len(self.margins_item1), len(self.margins_item2)))
            matching = np.random.rand(4, 4)
        else:
            if np.random.binomial(1, 1-self.alpha):
                # Computing the upper_conf for each pair of margins, along with the corresponding matching matrix.
                # Note that the conversion rates for the two items are the learned ones (empirical means plus confidence)
                upper_conf = np.zeros((len(self.margins_item1), len(self.margins_item2)))
                matching = np.zeros((len(self.margins_item1), len(self.margins_item2), 4, 4))

                for margin1 in range(len(self.margins_item1)):
                    for margin2 in range(len(self.margins_item2)):
                        daily_promos = (self.promo_fractions * sum(self.daily_customers * (self.empirical_means_item1[margin1] + self.confidence_item1[margin1]))).astype(int)
                        revenue_item2, matching[margin1][margin2] = lp.matching_lp(self.margins_item2[margin2], self.discounts, (self.empirical_means_item2[margin2] + self.confidence_item2[margin2]),
                                                                                   daily_promos, (self.daily_customers * (self.empirical_means_item1[margin1] + self.confidence_item1[margin1])).astype(int))
                        upper_conf[margin1][margin2] = self.margins_item1[margin1] * (self.daily_customers * (self.empirical_means_item1[margin1] + self.confidence_item1[margin1])).sum() + revenue_item2

                # Selecting the indices of the maximum of the upper_conf matrix (breaking ties randomly)
                # and the corresponding matching matrix
                arm_flat = np.argmax(np.random.random(upper_conf.shape) * (upper_conf == np.amax(upper_conf, None, keepdims=True)), None)
                arm = np.unravel_index(arm_flat, upper_conf.shape)
                matching = matching[arm[0]][arm[1]]
            else:
                # Returning a random pulled arm and matching matrix
                arm = [np.random.randint(0, len(self.margins_item1)), np.random.randint(0, len(self.margins_item2))]
                matching = np.random.rand(4, 4)

        return arm, matching

    # Pulled_arm contains the indices of the two pulled arms (pulled_arm[0] and pulled_arm[1]).
    # Reward contains the array of conversion rates for the first item (reward[0]), the matrix of conversion rates for
    # the second item (reward[1]) obtained from the environment.
    # Change Detection mechanism is put in place.
    def update(self, pulled_arm, reward):
        self.t += 1

        # Change detection and empirical means item 1
        for class_type in range(4):
            if self.change_detection_item1[pulled_arm[0]][class_type].update(reward[0][class_type]):
                self.detections_item1[pulled_arm[0]][class_type].append(self.t)
                self.valid_rewards_per_arms_item1[pulled_arm[0]][class_type] = []
                self.change_detection_item1[pulled_arm[0]][class_type].reset()
            self.valid_rewards_per_arms_item1[pulled_arm[0]][class_type].append(reward[0][class_type])
            self.empirical_means_item1[pulled_arm[0]][class_type] = np.mean(self.valid_rewards_per_arms_item1[pulled_arm[0]][class_type])

        # Confidence item 1
        total_valid_samples = 0
        for margin1 in range(len(self.margins_item1)):
            for class_type in range(4):
                total_valid_samples += len(self.valid_rewards_per_arms_item1[margin1][class_type])
        for margin1 in range(len(self.margins_item1)):
            for class_type in range(4):
                number_pulled = max(1, len(self.valid_rewards_per_arms_item1[margin1][class_type]))
                self.confidence_item1[margin1][class_type] = (2 * np.log(total_valid_samples) / number_pulled) ** 0.5

        # Change detection and empirical means item 2
        for promo_type in range(4):
            for class_type in range(4):
                if self.change_detection_item2[pulled_arm[1]][promo_type][class_type].update(reward[1][promo_type][class_type]):
                    self.detections_item2[pulled_arm[1]][promo_type][class_type].append(self.t)
                    self.valid_rewards_per_arms_item2[pulled_arm[1]][promo_type][class_type] = []
                    self.change_detection_item2[pulled_arm[1]][promo_type][class_type].reset()
                self.valid_rewards_per_arms_item2[pulled_arm[1]][promo_type][class_type].append(reward[1][promo_type][class_type])
                self.empirical_means_item2[pulled_arm[1]][promo_type][class_type] = np.mean(self.valid_rewards_per_arms_item2[pulled_arm[1]][promo_type][class_type])

        # Confidence item 2
        total_valid_samples = 0
        for margin2 in range(len(self.margins_item2)):
            for promo_type in range(4):
                for class_type in range(4):
                    total_valid_samples += len(self.valid_rewards_per_arms_item2[margin2][promo_type][class_type])
        for margin2 in range(len(self.margins_item2)):
            for promo_type in range(4):
                for class_type in range(4):
                    number_pulled = max(1, len(self.valid_rewards_per_arms_item2[margin2][promo_type][class_type]))
                    self.confidence_item2[margin2][promo_type][class_type] = (2 * np.log(total_valid_samples) / number_pulled) ** 0.5


# Cumulative Sum class, used by the Change Detection UCB1 Bandit
class CUMSUM:
    def __init__(self, M, eps, h):
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample):
        self.t += 1
        if self.t <= self.M:
            self.reference += sample/self.M
            return 0
        else:
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, np.max(self.g_plus + s_plus))
            self.g_minus = max(0, np.max(self.g_minus + s_minus))
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.t = 0
        self.g_minus = 0
        self.g_plus = 0
