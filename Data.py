from numpy.random import normal, binomial
from Form.Form import *

np.random.seed(1234)


class Data:
    def __init__(self):
        form = Form()

        # Daily number of customers per class (array of length 4)
        self.daily_customers = np.array([int(normal(form.n[0], 12)),
                                         int(normal(form.n[1], 14)),
                                         int(normal(form.n[2], 16)),
                                         int(normal(form.n[3], 17))])

        # Candidate prices and margins for item 1
        self.prices_item1 = np.array([150, 200, 300, 400, 450])
        self.margins_item1 = np.array([50, 75, 100, 125, 150])

        # Probability that a customer of a class buys the first item, according to its price
        # 4x5 matrix -> rows: customer class (class1, class2, class3, class4); columns: candidate prices
        self.conversion_rates_item1 = np.array([[0.72, 0.59, form.i1_param[0], 0.42, 0.23],
                                                [0.57, 0.44, form.i1_param[1], 0.29, 0.13],
                                                [0.62, 0.48, form.i1_param[2], 0.36, 0.17],
                                                [0.59, 0.44, form.i1_param[3], 0.31, 0.15]])

        # Candidate prices and margins for item 2
        self.prices_item2 = np.array([40, 50, 60])
        self.margins_item2 = np.array([15, 20, 25])

        # Probability that a customer of a class buys the second item given that he bought the first and has a promo
        # 4x4 matrix -> rows: promo (P0, P1, P2, P3); columns: customer class (class1, class2, class3, class4)
        # Each one of these matrix corresponds to a candidate price for item 2
        conversion_rates_item2_low_price = np.array([
                            # Promo code P0 - Class1, Class2, Class3, Class4
                            [(binomial(self.daily_customers[0], form.i21_param[0][0]) + 20) / self.daily_customers[0],
                             (binomial(self.daily_customers[1], form.i21_param[0][1]) + 30) / self.daily_customers[1],
                             (binomial(self.daily_customers[2], form.i21_param[0][2]) + 5) / self.daily_customers[2],
                             (binomial(self.daily_customers[3], form.i21_param[0][3]) + 2) / self.daily_customers[3]],
                            # Promo code P1 - Class1, Class2, Class3, Class4
                            [(binomial(self.daily_customers[0], form.i21_param[1][0]) + 19) / self.daily_customers[0],
                             (binomial(self.daily_customers[1], form.i21_param[1][1]) + 21) / self.daily_customers[1],
                             (binomial(self.daily_customers[2], form.i21_param[1][2]) + 0) / self.daily_customers[2],
                             (binomial(self.daily_customers[3], form.i21_param[1][3]) + 0) / self.daily_customers[3]],
                            # Promo code P2 - Class1, Class2, Class3, Class4
                            [(binomial(self.daily_customers[0], form.i21_param[2][0]) + 7) / self.daily_customers[0],
                             (binomial(self.daily_customers[1], form.i21_param[2][1]) + 6) / self.daily_customers[1],
                             (binomial(self.daily_customers[2], form.i21_param[2][2]) + 1) / self.daily_customers[2],
                             (binomial(self.daily_customers[3], form.i21_param[2][3]) + 2) / self.daily_customers[3]],
                            # Promo code P3 - Class1, Class2, Class3, Class4
                            [(binomial(self.daily_customers[0], form.i21_param[3][0]) + 10) / self.daily_customers[0],
                             (binomial(self.daily_customers[1], form.i21_param[3][1]) + 17) / self.daily_customers[1],
                             (binomial(self.daily_customers[2], form.i21_param[3][2]) + 4) / self.daily_customers[2],
                             (binomial(self.daily_customers[3], form.i21_param[3][3]) + 0) / self.daily_customers[3]]
                        ])

        conversion_rates_item2_middle_price = np.array([
                                    # Promo code P0 - Class1, Class2, Class3, Class4
                                    [binomial(self.daily_customers[0], form.i21_param[0][0]) / self.daily_customers[0],
                                     binomial(self.daily_customers[1], form.i21_param[0][1]) / self.daily_customers[1],
                                     binomial(self.daily_customers[2], form.i21_param[0][2]) / self.daily_customers[2],
                                     binomial(self.daily_customers[3], form.i21_param[0][3]) / self.daily_customers[3]],
                                    # Promo code P1 - Class1, Class2, Class3, Class4
                                    [binomial(self.daily_customers[0], form.i21_param[1][0]) / self.daily_customers[0],
                                     binomial(self.daily_customers[1], form.i21_param[1][1]) / self.daily_customers[1],
                                     binomial(self.daily_customers[2], form.i21_param[1][2]) / self.daily_customers[2],
                                     binomial(self.daily_customers[3], form.i21_param[1][3]) / self.daily_customers[3]],
                                    # Promo code P2 - Class1, Class2, Class3, Class4
                                    [binomial(self.daily_customers[0], form.i21_param[2][0]) / self.daily_customers[0],
                                     binomial(self.daily_customers[1], form.i21_param[2][1]) / self.daily_customers[1],
                                     binomial(self.daily_customers[2], form.i21_param[2][2]) / self.daily_customers[2],
                                     binomial(self.daily_customers[3], form.i21_param[2][3]) / self.daily_customers[3]],
                                    # Promo code P3 - Class1, Class2, Class3, Class4
                                    [binomial(self.daily_customers[0], form.i21_param[3][0]) / self.daily_customers[0],
                                     binomial(self.daily_customers[1], form.i21_param[3][1]) / self.daily_customers[1],
                                     binomial(self.daily_customers[2], form.i21_param[3][2]) / self.daily_customers[2],
                                     binomial(self.daily_customers[3], form.i21_param[3][3]) / self.daily_customers[3]]
                                ])

        conversion_rates_item2_high_price = np.array([
                            # Promo code P0 - Class1, Class2, Class3, Class4
                            [(binomial(self.daily_customers[0], form.i21_param[0][0]) - 15) / self.daily_customers[0],
                             (binomial(self.daily_customers[1], form.i21_param[0][1]) - 27) / self.daily_customers[1],
                             (binomial(self.daily_customers[2], form.i21_param[0][2]) - 1) / self.daily_customers[2],
                             (binomial(self.daily_customers[3], form.i21_param[0][3]) - 2) / self.daily_customers[3]],
                            # Promo code P1 - Class1, Class2, Class3, Class4
                            [(binomial(self.daily_customers[0], form.i21_param[1][0]) - 10) / self.daily_customers[0],
                             (binomial(self.daily_customers[1], form.i21_param[1][1]) - 23) / self.daily_customers[1],
                             (binomial(self.daily_customers[2], form.i21_param[1][2]) - 2) / self.daily_customers[2],
                             (binomial(self.daily_customers[3], form.i21_param[1][3]) - 5) / self.daily_customers[3]],
                            # Promo code P2 - Class1, Class2, Class3, Class4
                            [(binomial(self.daily_customers[0], form.i21_param[2][0]) - 14) / self.daily_customers[0],
                             (binomial(self.daily_customers[1], form.i21_param[2][1]) - 8) / self.daily_customers[1],
                             (binomial(self.daily_customers[2], form.i21_param[2][2]) - 3) / self.daily_customers[2],
                             (binomial(self.daily_customers[3], form.i21_param[2][3]) - 7) / self.daily_customers[3]],
                            # Promo code P3 - Class1, Class2, Class3, Class4
                            [(binomial(self.daily_customers[0], form.i21_param[3][0]) - 9) / self.daily_customers[0],
                             (binomial(self.daily_customers[1], form.i21_param[3][1]) - 7) / self.daily_customers[1],
                             (binomial(self.daily_customers[2], form.i21_param[3][2]) - 4) / self.daily_customers[2],
                             (binomial(self.daily_customers[3], form.i21_param[3][3]) - 0) / self.daily_customers[3]]
                        ])

        self.conversion_rates_item2 = np.array([conversion_rates_item2_low_price,
                                                conversion_rates_item2_middle_price,
                                                conversion_rates_item2_high_price])
        self.conversion_rates_item2 = np.clip(self.conversion_rates_item2, 0, 1)
