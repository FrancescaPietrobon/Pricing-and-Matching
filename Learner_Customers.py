import numpy as np
np.random.seed(1234)


class Learner_Customers:
    def __init__(self):
        self.t = 0

    def update_daily_customers(self, empirical_means, sample):
        self.t += 1
        return (empirical_means * (self.t - 1) + sample) / self.t
