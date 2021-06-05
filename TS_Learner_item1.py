from Learner import *
np.random.seed(1234)


class TS_Learner_item1():
    def __init__(self, n_arms, daily_customers, margins, reward_item2):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])
        self.beta_parameters = np.ones((4, n_arms, 2))
        self.daily_customers = daily_customers
        self.margins = margins
        self.reward_item2 = reward_item2

    def pull_arm(self):
        beta = np.random.beta(self.beta_parameters[:, :, 0], self.beta_parameters[:, :, 1])
        value = (self.margins * np.dot(self.daily_customers, beta)) + np.dot(self.daily_customers * self.reward_item2, beta)
        arm = np.random.choice(np.where(value == value.max())[0])
        return arm

    def update(self, pulled_arm, reward, revenue):
        self.t += 1
        self.collected_rewards = np.append(self.collected_rewards, revenue)
        self.beta_parameters[:, pulled_arm, 0] = self.beta_parameters[:, pulled_arm, 0] + reward
        self.beta_parameters[:, pulled_arm, 1] = self.beta_parameters[:, pulled_arm, 1] + 1.0 - reward
