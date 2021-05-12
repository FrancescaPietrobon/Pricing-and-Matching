from Learner import *
np.random.seed(1234)


class TS_Learner_item2(Learner):
    def __init__(self, n_arms, price, discounts, weights):
        super().__init__(n_arms)
        self.collected_rewards_matrix = np.zeros(n_arms)
        self.beta_parameters = np.ones((4, n_arms, 2))
        self.price = price
        self.discounts = discounts
        self.weights = weights

    def pull_arm(self):
        beta = np.random.beta(self.beta_parameters[:, :, 0], self.beta_parameters[:, :, 1])
        value = (self.price * np.dot(self.discounts * self.weights, beta))
        arm = np.random.choice(np.where(value == value.max())[0])
        return arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.collected_rewards_matrix = np.append(self.collected_rewards_matrix, self.price * self.discounts * self.weights * reward)
        self.collected_rewards = np.append(self.collected_rewards, np.sum(self.price * self.discounts * self.weights * reward))
        self.beta_parameters[:, pulled_arm, 0] = self.beta_parameters[:, pulled_arm, 0] + reward
        self.beta_parameters[:, pulled_arm, 1] = self.beta_parameters[:, pulled_arm, 1] + 1.0 - reward

    def get_mean_collected_rewards_matrix(self):
        return np.mean(self.collected_rewards_matrix, axis=0)
