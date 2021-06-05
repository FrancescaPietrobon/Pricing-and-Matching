import numpy as np
np.random.seed(1234)


class UCB1_item1():
    def __init__(self, n_arms, daily_customers, margins, reward_item2):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])
        self.empirical_means = np.zeros((4, n_arms))
        self.confidence = np.zeros(n_arms)
        self.daily_customers = daily_customers
        self.margins = margins
        self.reward_item2 = reward_item2

    def pull_arm(self):
        if self.t < self.n_arms:
            arm = self.t
        else:
            upper_bound = (self.margins * np.dot(self.daily_customers, (self.empirical_means + self.confidence))) + \
                          np.dot(self.daily_customers * self.reward_item2, (self.empirical_means + self.confidence))
            arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
        return arm

    def update(self, pulled_arm, reward, revenue):
        self.t += 1
        self.collected_rewards = np.append(self.collected_rewards, revenue)
        self.empirical_means[:, pulled_arm] = (self.empirical_means[:, pulled_arm] * (self.t-1) + reward) / self.t
        for a in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[a]))
            self.confidence[a] = (2*np.log(self.t) / number_pulled)**0.5
        self.rewards_per_arm[pulled_arm].append(reward)
