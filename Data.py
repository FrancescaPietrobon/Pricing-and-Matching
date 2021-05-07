import numpy as np
from numpy.random import normal, binomial
from Form import *

np.random.seed(1234)


class Data:
    def __init__(self):
        data = Form()

        # Daily number of customers per class = Gaussian TODO: are sigmas correct?
        self.daily_customers = np.array([int(normal(data.get_n(1), 12)),
                                         int(normal(data.get_n(2), 14)),
                                         int(normal(data.get_n(3), 16)),
                                         int(normal(data.get_n(4), 17))])

        # Probability that a customer of a class buys the second item given the first + each promo
        # rows: promo code (0: P0, 1: P1, 2: P2, 3: P3)
        # columns: customer group (0: group1, 1: group2, 2: group3, 3: group4)
        self.conversion_rates_item21 = np.array([  # Promo code P0
            [binomial(self.daily_customers[0], data.get_i21_p0_param(1)) / self.daily_customers[0],
             binomial(self.daily_customers[1], data.get_i21_p0_param(2)) / self.daily_customers[1],
             binomial(self.daily_customers[2], data.get_i21_p0_param(3)) / self.daily_customers[2],
             binomial(self.daily_customers[3], data.get_i21_p0_param(4)) / self.daily_customers[3]],
            # Promo code P1
            [binomial(self.daily_customers[0], data.get_i21_p1_param(1)) / self.daily_customers[0],
             binomial(self.daily_customers[1], data.get_i21_p1_param(2)) / self.daily_customers[1],
             binomial(self.daily_customers[2], data.get_i21_p1_param(3)) / self.daily_customers[2],
             binomial(self.daily_customers[3], data.get_i21_p1_param(4)) / self.daily_customers[3]],
            # Promo code P2
            [binomial(self.daily_customers[0], data.get_i21_p2_param(1)) / self.daily_customers[0],
             binomial(self.daily_customers[1], data.get_i21_p2_param(2)) / self.daily_customers[1],
             binomial(self.daily_customers[2], data.get_i21_p2_param(3)) / self.daily_customers[2],
             binomial(self.daily_customers[3], data.get_i21_p2_param(4)) / self.daily_customers[3]],
            # Promo code P3
            [binomial(self.daily_customers[0], data.get_i21_p3_param(1)) / self.daily_customers[0],
             binomial(self.daily_customers[1], data.get_i21_p3_param(2)) / self.daily_customers[1],
             binomial(self.daily_customers[2], data.get_i21_p3_param(3)) / self.daily_customers[2],
             binomial(self.daily_customers[3], data.get_i21_p3_param(4)) / self.daily_customers[3]]
        ])

    def get_daily_customers(self):
        return self.daily_customers

    def get_conversion_rates_item21(self):
        return self.conversion_rates_item21
