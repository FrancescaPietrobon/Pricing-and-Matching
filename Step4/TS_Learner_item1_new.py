from Learner import *
np.random.seed(1234)


class TS_Learner_item1_new(Learner):
    def __init__(self, n_arms, daily_customers, prices, reward_item2):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((4, n_arms, 2))
        self.daily_customers = daily_customers
        self.prices = prices
        self.reward_item2 = reward_item2

    def pull_arm(self):
        beta = np.random.beta(self.beta_parameters[:, :, 0], self.beta_parameters[:, :, 1])
        value = (self.prices * np.dot(self.daily_customers, beta)) + np.dot(self.daily_customers * self.reward_item2, beta)
        arm = np.random.choice(np.where(value == value.max())[0])
        return arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.collected_rewards = np.append(self.collected_rewards, np.sum((self.prices[pulled_arm] + self.reward_item2) * self.daily_customers * reward))
        self.beta_parameters[:, pulled_arm, 0] = self.beta_parameters[:, pulled_arm, 0] + reward
        self.beta_parameters[:, pulled_arm, 1] = self.beta_parameters[:, pulled_arm, 1] + 1.0 - reward

    def update_reward_item2(self, updated_reward_item2):
        self.reward_item2 = updated_reward_item2

    def update_daily_customers(self, daily_customers_empirical_means):
        self.daily_customers = daily_customers_empirical_means