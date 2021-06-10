import numpy as np
np.random.seed(1234)


class TS_Item1():
    def __init__(self, n_arms, margins_item1, selected_margin_item2, conversion_rates_item2, weights, daily_customers, discounts):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])
        self.beta_parameters = np.ones((4, n_arms, 2))

        self.margins_item1 = margins_item1
        self.margin_item2 = selected_margin_item2
        self.conversion_rates_item2 = conversion_rates_item2
        self.weights = weights
        self.daily_customers = daily_customers
        self.discounts = discounts

    def pull_arm(self):
        beta = np.random.beta(self.beta_parameters[:, :, 0], self.beta_parameters[:, :, 1])

        reward_item1 = (self.margins_item1 * np.dot(self.daily_customers, beta))
        reward_item2 = np.zeros(4)
        for class_type in range(4):
            reward_item2[class_type] = self.margin_item2 * self.daily_customers[class_type] * ((1 - self.discounts) *
                                       self.conversion_rates_item2[:, class_type] * self.weights[:, class_type]).sum()
        reward_item2 = np.dot(reward_item2, beta)

        value = reward_item1 + reward_item2
        arm = np.random.choice(np.where(value == value.max())[0])
        return arm

    def update(self, pulled_arm, reward, revenue):
        self.t += 1
        self.collected_rewards = np.append(self.collected_rewards, revenue)
        self.beta_parameters[:, pulled_arm, 0] = self.beta_parameters[:, pulled_arm, 0] + reward
        self.beta_parameters[:, pulled_arm, 1] = self.beta_parameters[:, pulled_arm, 1] + 1.0 - reward
