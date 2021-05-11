from Learner import Learner
import numpy as np
np.random.seed(1234)


class UCB1_item1_new(Learner):
    def __init__(self, n_arms, daily_customers, prices, reward_item2):
        super().__init__(n_arms)
        self.empirical_means = np.zeros([4, n_arms])
        self.confidence = np.zeros(n_arms)
        self.daily_customers = daily_customers
        self.prices = prices
        self.reward_item2 = reward_item2

    def pull_arm(self):
        if self.t < self.n_arms:
            arm = self.t
        else:
            upper_bound = (self.prices * np.dot(self.daily_customers, (self.empirical_means + self.confidence))) + np.dot(self.daily_customers * self.reward_item2, (self.empirical_means + self.confidence))
            arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
        return arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.collected_rewards = np.append(self.collected_rewards, np.sum((self.prices[pulled_arm] + self.reward_item2) * self.daily_customers * reward))
        self.empirical_means[:, pulled_arm] = (self.empirical_means[:, pulled_arm] * (self.t-1) + reward) / self.t
        for a in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[a]))
            self.confidence[a] = (2*np.log(self.t) / number_pulled)**0.5
        self.rewards_per_arm[pulled_arm].append(reward)

    def update_reward_item2(self, updated_reward_item2):
        self.reward_item2 = updated_reward_item2

    def update_daily_customers(self, daily_customers_empirical_means):
        self.daily_customers = daily_customers_empirical_means
