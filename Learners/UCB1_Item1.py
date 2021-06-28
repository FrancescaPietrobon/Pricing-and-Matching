import numpy as np
np.random.seed(1234)


class UCB1_Item1():
    def __init__(self, n_arms, margins_item1, selected_margin_item2, conversion_rates_item2, weights, daily_customers, discounts):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])
        self.empirical_means = np.zeros((4, n_arms))
        self.confidence = np.full(n_arms, np.inf)

        self.margins_item1 = margins_item1
        self.margin_item2 = selected_margin_item2
        self.conversion_rates_item2 = conversion_rates_item2
        self.weights = weights
        self.daily_customers = daily_customers
        self.discounts = discounts

    def pull_arm(self):
        if self.t < self.n_arms:
            arm = self.t
        else:
            # Computing the upper bound as the total revenue obtained during the round (day).
            # Note that the conversion rates for the first item are the learned ones (empirical means plus confidence).
            revenue_item1 = self.margins_item1 * np.dot(self.daily_customers, (self.empirical_means + self.confidence))
            revenue_item2 = np.zeros(4)
            for class_type in range(4):
                revenue_item2[class_type] = self.margin_item2 * self.daily_customers[class_type] * ((1 - self.discounts) *
                                            self.conversion_rates_item2[:, class_type] * self.weights[:, class_type]).sum()
            revenue_item2 = np.dot(revenue_item2, (self.empirical_means + self.confidence))

            upper_bound = revenue_item1 + revenue_item2
            arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
        return arm

    # Pulled_arm is the arm pulled in the above method.
    # Reward is the array of conversion rates for the first item, obtained from the environment.
    # Revenue is the total revenue for the day, obtained from the environment.
    def update(self, pulled_arm, reward, revenue):
        self.t += 1
        self.collected_rewards = np.append(self.collected_rewards, revenue)
        self.empirical_means[:, pulled_arm] = (self.empirical_means[:, pulled_arm] * (self.t-1) + reward) / self.t
        for a in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[a]))
            self.confidence[a] = (2*np.log(self.t) / number_pulled)**0.5
        self.rewards_per_arm[pulled_arm].append(reward)
