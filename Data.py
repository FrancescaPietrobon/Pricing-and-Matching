from numpy.random import normal, binomial
from Form import *

np.random.seed(1234)


class Data:
    def __init__(self):
        form = Form()

        # Daily number of customers per class = Gaussian TODO: are sigmas correct?
        self.daily_customers = np.array([int(normal(form.get_n()[0], 12)),
                                         int(normal(form.get_n()[1], 14)),
                                         int(normal(form.get_n()[2], 16)),
                                         int(normal(form.get_n()[3], 17))])

        # Probability that a customer of a class buys the second item given that he bought the first and has a promo
        # 4x4 matrix -> rows: promo (P0, P1, P2, P3); columns: customer class (class1, class2, class3, class4)
        self.conversion_rates_item21 = np.array([
            # Promo code P0 - Class1, Class2, Class3, Class4
            [binomial(self.daily_customers[0], form.get_i21_param()[0][0]) / self.daily_customers[0],
             binomial(self.daily_customers[1], form.get_i21_param()[0][1]) / self.daily_customers[1],
             binomial(self.daily_customers[2], form.get_i21_param()[0][2]) / self.daily_customers[2],
             binomial(self.daily_customers[3], form.get_i21_param()[0][3]) / self.daily_customers[3]],
            # Promo code P1 - Class1, Class2, Class3, Class4
            [binomial(self.daily_customers[0], form.get_i21_param()[1][0]) / self.daily_customers[0],
             binomial(self.daily_customers[1], form.get_i21_param()[1][1]) / self.daily_customers[1],
             binomial(self.daily_customers[2], form.get_i21_param()[1][2]) / self.daily_customers[2],
             binomial(self.daily_customers[3], form.get_i21_param()[1][3]) / self.daily_customers[3]],
            # Promo code P2 - Class1, Class2, Class3, Class4
            [binomial(self.daily_customers[0], form.get_i21_param()[2][0]) / self.daily_customers[0],
             binomial(self.daily_customers[1], form.get_i21_param()[2][1]) / self.daily_customers[1],
             binomial(self.daily_customers[2], form.get_i21_param()[2][2]) / self.daily_customers[2],
             binomial(self.daily_customers[3], form.get_i21_param()[2][3]) / self.daily_customers[3]],
            # Promo code P3 - Class1, Class2, Class3, Class4
            [binomial(self.daily_customers[0], form.get_i21_param()[3][0]) / self.daily_customers[0],
             binomial(self.daily_customers[1], form.get_i21_param()[3][1]) / self.daily_customers[1],
             binomial(self.daily_customers[2], form.get_i21_param()[3][2]) / self.daily_customers[2],
             binomial(self.daily_customers[3], form.get_i21_param()[3][3]) / self.daily_customers[3]]
        ])

        # Candidate prices for item 1
        # TODO: define item1 and item 2 in this class instead of simulator, to take the price (€300) from here
        self.prices_item1 = np.array([50, 100, 150, 200, 300, 400, 450, 500, 550])

        # Candidate conversion rates for item 1 (row: customer class; column: price item 1)
        self.conversion_rates_item1 = np.array([[0.90, 0.84, 0.72, 0.59, form.get_i1_param()[0], 0.42, 0.23, 0.13, 0.07],
                                                [0.87, 0.75, 0.57, 0.44, form.get_i1_param()[1], 0.29, 0.13, 0.10, 0.02],
                                                [0.89, 0.78, 0.62, 0.48, form.get_i1_param()[2], 0.36, 0.17, 0.12, 0.05],
                                                [0.88, 0.78, 0.59, 0.44, form.get_i1_param()[3], 0.31, 0.15, 0.13, 0.03]])

    def get_daily_customers(self):
        return self.daily_customers

    def get_conversion_rates_item21(self):
        return self.conversion_rates_item21

    def get_prices_item1(self):
        return self.prices_item1

    def get_conversion_rates_item1(self):
        return self.conversion_rates_item1
