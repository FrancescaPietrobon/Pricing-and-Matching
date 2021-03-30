import numpy as np
from Day import *
from Group import *
from CustomerData import *
from Data import *


class Simulator:
    def __init__(self):
        self.group1 = Group(1)
        self.group2 = Group(2)
        self.group3 = Group(3)
        self.group4 = Group(4)
        self.day = Day(1, self.group1, self.group2, self.group3, self.group4)

    # TODO: import data from Google module

    def simulation_step_1(self):
        data = Data()

        p0_frac = 0.7
        p1_frac = 0.2
        p2_frac = 0.07
        p3_frac = 0.03

        # Number of customers per class = potentially different Gaussian distributions TODO: are sigmas correct?
        c1_daily = int(np.random.normal(data.get_n(1), 12))
        c2_daily = int(np.random.normal(data.get_n(2), 14))
        c3_daily = int(np.random.normal(data.get_n(3), 16))
        c4_daily = int(np.random.normal(data.get_n(4), 17))

        p0_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p0_frac)
        p1_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p1_frac)
        p2_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p2_frac)
        p3_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p3_frac)

        # Probability that a customer of a class buys the first item alone = Binomial
        c1_i1 = np.random.binomial(c1_daily, data.get_i1_param(1)) / c1_daily
        c2_i1 = np.random.binomial(c2_daily, data.get_i1_param(2)) / c2_daily
        c3_i1 = np.random.binomial(c3_daily, data.get_i1_param(3)) / c3_daily
        c4_i1 = np.random.binomial(c4_daily, data.get_i1_param(4)) / c4_daily

        # Probability that a customer of a class buys the second item alone = Binomial TODO: values by hand are ok?
        c1_i2 = np.random.binomial(c1_daily, 0.276) / c1_daily              # Class 1: N = 76, p = 0.276
        c2_i2 = np.random.binomial(c2_daily, 0.421) / c2_daily            # Class 2: N = 133, p = 0.421
        c3_i2 = np.random.binomial(c3_daily, 0.358) / c3_daily            # Class 3: N = 107, p = 0.358
        c4_i2 = np.random.binomial(c4_daily, 0.452) / c4_daily              # Class 4: N = 93, p = 0.452

        # Probability that a customer of a class buys the second item given the first + P0 = Binomial
        c1_i21_p0 = np.random.binomial(c1_daily, data.get_i21_p0_param(1)) / c1_daily
        c2_i21_p0 = np.random.binomial(c2_daily, data.get_i21_p0_param(2)) / c2_daily
        c3_i21_p0 = np.random.binomial(c3_daily, data.get_i21_p0_param(3)) / c3_daily
        c4_i21_p0 = np.random.binomial(c4_daily, data.get_i21_p0_param(4)) / c4_daily

        # Probability that a customer of a class buys the second item given the first + P1 = Binomial
        c1_i21_p1 = np.random.binomial(c1_daily, data.get_i21_p1_param(1)) / c1_daily
        c2_i21_p1 = np.random.binomial(c2_daily, data.get_i21_p1_param(2)) / c2_daily
        c3_i21_p1 = np.random.binomial(c3_daily, data.get_i21_p1_param(3)) / c3_daily
        c4_i21_p1 = np.random.binomial(c4_daily, data.get_i21_p1_param(4)) / c4_daily

        # Probability that a customer of a class buys the second item given the first + P2 = Binomial
        c1_i21_p2 = np.random.binomial(c1_daily, data.get_i21_p2_param(1)) / c1_daily
        c2_i21_p2 = np.random.binomial(c2_daily, data.get_i21_p2_param(2)) / c2_daily
        c3_i21_p2 = np.random.binomial(c3_daily, data.get_i21_p2_param(3)) / c3_daily
        c4_i21_p2 = np.random.binomial(c4_daily, data.get_i21_p2_param(4)) / c4_daily

        # Probability that a customer of a class buys the second item given the first + P3 = Binomial
        c1_i21_p3 = np.random.binomial(c1_daily, data.get_i21_p3_param(1)) / c1_daily
        c2_i21_p3 = np.random.binomial(c2_daily, data.get_i21_p3_param(2)) / c2_daily
        c3_i21_p3 = np.random.binomial(c3_daily, data.get_i21_p3_param(3)) / c3_daily
        c4_i21_p3 = np.random.binomial(c4_daily, data.get_i21_p3_param(4)) / c4_daily

        # For each class of customers, we create the different CustomerData objects according to the above distributions
        # Every time we give a promo to the customer, we decrease the number of coupons available

        for customer in range(c1_daily):
            customer_data = CustomerData(customer+1, self.group1.get_number())
            if np.random.binomial(1, c1_i1) == 1:                               # If customer buys item 1 alone
                promo = np.random.randint(1, 4)                                 # We can give it a promo (P1, P2, P3)
                give_promo = np.random.binomial()                               # With a 50% chance it gets P0 and with a 50% chance it gets P1, P2 or P3 according to the previous result
                if give_promo == 0 and p0_num > 0:                              # It gets P0
                    customer_data.set_true_first_promo()
                    p0_num -= 1
                elif promo == 1 and p1_num > 0:                                 # It gets P1
                    customer_data.set_true_second_promo()
                    p1_num -= 1
                elif promo == 2 and p2_num > 0:                                # It gets P2
                    customer_data.set_true_third_promo()
                    p2_num -= 1
                elif promo == 3 and p3_num > 0:                                # It gets P3
                    customer_data.set_true_fourth_promo()
                    p3_num -= 1
                customer_data.set_true_first_purchase()
                self.day.add_customer_data(customer_data)
            elif np.random.binomial(1, c1_i2) == 1:
                customer_data.set_true_second_purchase()
                self.day.add_customer_data(customer_data)
            elif np.random.binomial(1, c1_i21_p0) == 1 and p0_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_first_promo()
                self.day.add_customer_data(customer_data)
                p0_num -= 1
            elif np.random.binomial(1, c1_i21_p1) == 1 and p1_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_second_promo()
                self.day.add_customer_data(customer_data)
                p1_num -= 1
            elif np.random.binomial(1, c1_i21_p2) == 1 and p2_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_third_promo()
                self.day.add_customer_data(customer_data)
                p2_num -= 1
            elif np.random.binomial(1, c1_i21_p3) == 1 and p3_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_fourth_promo()
                self.day.add_customer_data(customer_data)
                p3_num -= 1

        for customer in range(c2_daily):
            customer_data = CustomerData(customer+1, self.group2.get_number())
            if np.random.binomial(1, c2_i1) == 1:
                promo = np.random.randint(1, 4)  # We can give it a promo (P1, P2, P3)
                give_promo = np.random.binomial()  # With a 50% chance it gets P0 and with a 50% chance it gets P1, P2 or P3 according to the previous result
                if give_promo == 0 and p0_num > 0:  # It gets P0
                    customer_data.set_true_first_promo()
                    p0_num -= 1
                elif promo == 1 and p1_num > 0:  # It gets P1
                    customer_data.set_true_second_promo()
                    p1_num -= 1
                elif promo == 2 and p2_num > 0:  # It gets P2
                    customer_data.set_true_third_promo()
                    p2_num -= 1
                elif promo == 3 and p3_num > 0:  # It gets P3
                    customer_data.set_true_fourth_promo()
                    p3_num -= 1
                customer_data.set_true_first_purchase()
                self.day.add_customer_data(customer_data)
            elif np.random.binomial(1, c2_i2) == 1:
                customer_data.set_true_second_purchase()
                self.day.add_customer_data(customer_data)
            elif np.random.binomial(1, c2_i21_p0) == 1 and p0_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_first_promo()
                self.day.add_customer_data(customer_data)
                p0_num -= 1
            elif np.random.binomial(1, c2_i21_p1) == 1 and p1_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_second_promo()
                self.day.add_customer_data(customer_data)
                p1_num -= 1
            elif np.random.binomial(1, c2_i21_p2) == 1 and p2_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_third_promo()
                self.day.add_customer_data(customer_data)
                p2_num -= 1
            elif np.random.binomial(1, c2_i21_p3) == 1 and p3_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_fourth_promo()
                self.day.add_customer_data(customer_data)
                p3_num -= 1

        for customer in range(c3_daily):
            customer_data = CustomerData(customer+1, self.group3.get_number())
            if np.random.binomial(1, c3_i1) == 1:
                promo = np.random.randint(1, 4)  # We can give it a promo (P1, P2, P3)
                give_promo = np.random.binomial()  # With a 50% chance it gets P0 and with a 50% chance it gets P1, P2 or P3 according to the previous result
                if give_promo == 0 and p0_num > 0:  # It gets P0
                    customer_data.set_true_first_promo()
                    p0_num -= 1
                elif promo == 1 and p1_num > 0:  # It gets P1
                    customer_data.set_true_second_promo()
                    p1_num -= 1
                elif promo == 2 and p2_num > 0:  # It gets P2
                    customer_data.set_true_third_promo()
                    p2_num -= 1
                elif promo == 3 and p3_num > 0:  # It gets P3
                    customer_data.set_true_fourth_promo()
                    p3_num -= 1
                customer_data.set_true_first_purchase()
                self.day.add_customer_data(customer_data)
            elif np.random.binomial(1, c3_i2) == 1:
                customer_data.set_true_second_purchase()
                self.day.add_customer_data(customer_data)
            elif np.random.binomial(1, c3_i21_p0) == 1 and p0_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_first_promo()
                self.day.add_customer_data(customer_data)
                p0_num -= 1
            elif np.random.binomial(1, c3_i21_p1) == 1 and p1_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_second_promo()
                self.day.add_customer_data(customer_data)
                p1_num -= 1
            elif np.random.binomial(1, c3_i21_p2) == 1 and p2_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_third_promo()
                self.day.add_customer_data(customer_data)
                p2_num -= 1
            elif np.random.binomial(1, c3_i21_p3) == 1 and p3_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_fourth_promo()
                self.day.add_customer_data(customer_data)
                p3_num -= 1

        for customer in range(c4_daily):
            customer_data = CustomerData(customer+1, self.group4.get_number())
            if np.random.binomial(1, c4_i1) == 1:
                promo = np.random.randint(1, 4)  # We can give it a promo (P1, P2, P3)
                give_promo = np.random.binomial()  # With a 50% chance it gets P0 and with a 50% chance it gets P1, P2 or P3 according to the previous result
                if give_promo == 0 and p0_num > 0:  # It gets P0
                    customer_data.set_true_first_promo()
                    p0_num -= 1
                elif promo == 1 and p1_num > 0:  # It gets P1
                    customer_data.set_true_second_promo()
                    p1_num -= 1
                elif promo == 2 and p2_num > 0:  # It gets P2
                    customer_data.set_true_third_promo()
                    p2_num -= 1
                elif promo == 3 and p3_num > 0:  # It gets P3
                    customer_data.set_true_fourth_promo()
                    p3_num -= 1
                customer_data.set_true_first_purchase()
                self.day.add_customer_data(customer_data)
            elif np.random.binomial(1, c4_i2) == 1:
                customer_data.set_true_second_purchase()
                self.day.add_customer_data(customer_data)
            elif np.random.binomial(1, c4_i21_p0) == 1 and p0_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_first_promo()
                self.day.add_customer_data(customer_data)
                p0_num -= 1
            elif np.random.binomial(1, c4_i21_p1) == 1 and p1_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_second_promo()
                self.day.add_customer_data(customer_data)
                p1_num -= 1
            elif np.random.binomial(1, c4_i21_p2) == 1 and p2_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_third_promo()
                self.day.add_customer_data(customer_data)
                p2_num -= 1
            elif np.random.binomial(1, c4_i21_p3) == 1 and p3_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_fourth_promo()
                self.day.add_customer_data(customer_data)
                p3_num -= 1

    def get_day_step1(self):
        return self.day
