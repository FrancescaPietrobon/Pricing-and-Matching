class Learner_Customers:
    def __init__(self, initial_empirical_means):
        self.t = 0
        self.empirical_means = initial_empirical_means

    def update_daily_customers(self, sample):
        self.t += 1
        self.empirical_means = (self.empirical_means * (self.t - 1) + sample) / self.t
        return self.empirical_means
