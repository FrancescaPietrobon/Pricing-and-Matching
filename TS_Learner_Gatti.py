from Learner import *
np.random.seed(1234)


class TS_Learner_Gatti(Learner):
    def __init__(self, n_arms, daily_customers, prices, reward2Gatti):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((4, n_arms, 2))
        self.daily_customers = daily_customers
        self.prices = prices
        self.reward2Gatti = reward2Gatti

    def pull_arm(self):
        beta = np.random.beta(self.beta_parameters[:, :, 0], self.beta_parameters[:, :, 1])
        value = (self.prices + self.reward2Gatti) * np.dot(self.daily_customers, beta)
        arm = np.random.choice(np.where(value == value.max())[0])
        return arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.rewards_per_arm[pulled_arm].append(np.sum((self.prices[pulled_arm] + self.reward2Gatti[pulled_arm]) * self.daily_customers * reward))
        a = self.prices[pulled_arm]
        b = self.reward2Gatti[pulled_arm]
        c = self.daily_customers
        d = reward
        self.collected_rewards = np.append(self.collected_rewards,
                                           np.sum((self.prices[pulled_arm] + self.reward2Gatti[pulled_arm]) * self.daily_customers * reward))
        self.beta_parameters[:, pulled_arm, 0] = self.beta_parameters[:, pulled_arm, 0] + reward
        self.beta_parameters[:, pulled_arm, 1] = self.beta_parameters[:, pulled_arm, 1] + 1.0 - reward
