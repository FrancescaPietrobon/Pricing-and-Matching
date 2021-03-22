import numpy as np
from Day import *
from Group import *
from CustomerData import *


class Simulator:

    def simulation_step_1(self):
        day = Day(1)
        group1 = Group(1)
        group2 = Group(2)
        group3 = Group(3)
        group4 = Group(4)

        p0_frac = 0.7
        p1_frac = 0.2
        p2_frac = 0.07
        p3_frac = 0.03

        # Number of customers per class = potentially different Gaussian distributions
        c1_daily = int(np.random.normal(76, 12))                # Class 1: 76 customers, sigma 12
        c2_daily = int(np.random.normal(133, 14))               # Class 2: 133 customers, sigma 14
        c3_daily = int(np.random.normal(107, 16))               # Class 3: 107 customers, sigma 16
        c4_daily = int(np.random.normal(93, 17))                # Class 4: 93 customers, sigma 17

        p0_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p0_frac)
        p1_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p1_frac)
        p2_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p2_frac)
        p3_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p3_frac)

        # Probability that a customer of a class buys the first item alone = Binomial
        c1_i1 = np.random.binomial(76, 0.632) / 76              # Class 1: N = 76, p = 0.632
        c2_i1 = np.random.binomial(133, 0.114) / 133            # Class 2: N = 133, p = 0.114
        c3_i1 = np.random.binomial(107, 0.333) / 107            # Class 3: N = 107, p = 0.333
        c4_i1 = np.random.binomial(93, 0.713) / 93              # Class 4: N = 93, p = 0.713

        # Probability that a customer of a class buys the second item alone = Binomial
        c1_i2 = np.random.binomial(76, 0.276) / 76              # Class 1: N = 76, p = 0.276
        c2_i2 = np.random.binomial(133, 0.421) / 133            # Class 2: N = 133, p = 0.421
        c3_i2 = np.random.binomial(107, 0.358) / 107            # Class 3: N = 107, p = 0.358
        c4_i2 = np.random.binomial(93, 0.452) / 93              # Class 4: N = 93, p = 0.452

        # Probability that a customer of a class buys the second item given the first + P0 = Binomial
        c1_i21_p0 = np.random.binomial(76, 0.512) / 76           # Class 1: N = 76, p = 0.512
        c2_i21_p0 = np.random.binomial(133, 0.830) / 133         # Class 2: N = 133, p = 0.830
        c3_i21_p0 = np.random.binomial(107, 0.122) / 107         # Class 3: N = 107, p = 0.122
        c4_i21_p0 = np.random.binomial(93, 0.010) / 93           # Class 4: N = 93, p = 0.010

        # Probability that a customer of a class buys the second item given the first + P1 = Binomial
        c1_i21_p1 = np.random.binomial(76, 0.845) / 76           # Class 1: N = 76, p = 0.845
        c2_i21_p1 = np.random.binomial(133, 0.831) / 133         # Class 2: N = 133, p = 0.831
        c3_i21_p1 = np.random.binomial(107, 0.145) / 107         # Class 3: N = 107, p = 0.145
        c4_i21_p1 = np.random.binomial(93, 0.201) / 93           # Class 4: N = 93, p = 0.201

        # Probability that a customer of a class buys the second item given the first + P2 = Binomial
        c1_i21_p2 = np.random.binomial(76, 0.872) / 76           # Class 1: N = 76, p = 0.872
        c2_i21_p2 = np.random.binomial(133, 0.872) / 133         # Class 2: N = 133, p = 0.872
        c3_i21_p2 = np.random.binomial(107, 0.367) / 107         # Class 3: N = 107, p = 0.367
        c4_i21_p2 = np.random.binomial(93, 0.364) / 93           # Class 4: N = 93, p = 0.364

        # Probability that a customer of a class buys the second item given the first + P3 = Binomial
        c1_i21_p3 = np.random.binomial(76, 0.910) / 76           # Class 1: N = 76, p = 0.910
        c2_i21_p3 = np.random.binomial(133, 0.700) / 133         # Class 2: N = 133, p = 0.700
        c3_i21_p3 = np.random.binomial(107, 0.662) / 107         # Class 3: N = 107, p = 0.662
        c4_i21_p3 = np.random.binomial(93, 0.546) / 93           # Class 4: N = 93, p = 0.546


        # For each class of customers, we create the different CustomerData objects according to the previous probability distributions
        # Every time we give a promo to the customer, we decrease the number of coupons available

        for customer in range(c1_daily):
            customer_data = CustomerData(customer+1, group1.get_number())
            if np.random.binomial(1, c1_i1) == 1:                               # If customer buys item 1 alone
                promo = np.random.randint(1, 4)                                 # We can give it a promo (P1, P2, P3)
                give_promo = np.random.binomial()                               # With a 50% chance it gets P0 and with a 50% chance it gets P1, P2 or P3 according to the previous result
                if give_promo == 0 and p0_num > 0:                              # It gets P0
                    customer_data.set_true_first_promo()
                    p0_num -= p0_num
                elif promo == 1 and p1_num > 0:                                 # It gets P1
                    customer_data.set_true_second_promo()
                    p1_num -= p1_num
                elif promo == 2 and p2_num > 0:                                # It gets P2
                    customer_data.set_true_third_promo()
                    p2_num -= p2_num
                elif promo == 3 and p3_num > 0:                                # It gets P3
                    customer_data.set_true_fourth_promo()
                    p3_num -= p3_num
                customer_data.set_true_first_purchase()
                day.add_customer_data(customer_data)
            elif np.random.binomial(1, c1_i2) == 1:
                customer_data.set_true_second_purchase()
                day.add_customer_data(customer_data)
            elif np.random.binomial(1, c1_i21_p0) == 1 and p0_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_first_promo()
                day.add_customer_data(customer_data)
                p0_num -= p0_num
            elif np.random.binomial(1, c1_i21_p1) == 1 and p1_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_second_promo()
                day.add_customer_data(customer_data)
                p1_num -= p1_num
            elif np.random.binomial(1, c1_i21_p2) == 1 and p2_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_third_promo()
                day.add_customer_data(customer_data)
                p2_num -= p2_num
            elif np.random.binomial(1, c1_i21_p3) == 1 and p3_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_fourth_promo()
                day.add_customer_data(customer_data)
                p3_num -= p3_num

        for customer in range(c2_daily):
            customer_data = CustomerData(customer+1, group2.get_number())
            if np.random.binomial(1, c2_i1) == 1:
                promo = np.random.randint(1, 4)  # We can give it a promo (P1, P2, P3)
                give_promo = np.random.binomial()  # With a 50% chance it gets P0 and with a 50% chance it gets P1, P2 or P3 according to the previous result
                if give_promo == 0 and p0_num > 0:  # It gets P0
                    customer_data.set_true_first_promo()
                    p0_num -= p0_num
                elif promo == 1 and p1_num > 0:  # It gets P1
                    customer_data.set_true_second_promo()
                    p1_num -= p1_num
                elif promo == 2 and p2_num > 0:  # It gets P2
                    customer_data.set_true_third_promo()
                    p2_num -= p2_num
                elif promo == 3 and p3_num > 0:  # It gets P3
                    customer_data.set_true_fourth_promo()
                    p3_num -= p3_num
                customer_data.set_true_first_purchase()
                day.add_customer_data(customer_data)
            elif np.random.binomial(1, c2_i2) == 1:
                customer_data.set_true_second_purchase()
                day.add_customer_data(customer_data)
            elif np.random.binomial(1, c2_i21_p0) == 1 and p0_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_first_promo()
                day.add_customer_data(customer_data)
                p0_num -= p0_num
            elif np.random.binomial(1, c2_i21_p1) == 1 and p1_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_second_promo()
                day.add_customer_data(customer_data)
                p1_num -= p1_num
            elif np.random.binomial(1, c2_i21_p2) == 1 and p2_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_third_promo()
                day.add_customer_data(customer_data)
                p2_num -= p2_num
            elif np.random.binomial(1, c2_i21_p3) == 1 and p3_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_fourth_promo()
                day.add_customer_data(customer_data)
                p3_num -= p3_num

        for customer in range(c3_daily):
            customer_data = CustomerData(customer+1, group3.get_number())
            if np.random.binomial(1, c3_i1) == 1:
                promo = np.random.randint(1, 4)  # We can give it a promo (P1, P2, P3)
                give_promo = np.random.binomial()  # With a 50% chance it gets P0 and with a 50% chance it gets P1, P2 or P3 according to the previous result
                if give_promo == 0 and p0_num > 0:  # It gets P0
                    customer_data.set_true_first_promo()
                    p0_num -= p0_num
                elif promo == 1 and p1_num > 0:  # It gets P1
                    customer_data.set_true_second_promo()
                    p1_num -= p1_num
                elif promo == 2 and p2_num > 0:  # It gets P2
                    customer_data.set_true_third_promo()
                    p2_num -= p2_num
                elif promo == 3 and p3_num > 0:  # It gets P3
                    customer_data.set_true_fourth_promo()
                    p3_num -= p3_num
                customer_data.set_true_first_purchase()
                day.add_customer_data(customer_data)
            elif np.random.binomial(1, c3_i2) == 1:
                customer_data.set_true_second_purchase()
                day.add_customer_data(customer_data)
            elif np.random.binomial(1, c3_i21_p0) == 1 and p0_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_first_promo()
                day.add_customer_data(customer_data)
                p0_num -= p0_num
            elif np.random.binomial(1, c3_i21_p1) == 1 and p1_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_second_promo()
                day.add_customer_data(customer_data)
                p1_num -= p1_num
            elif np.random.binomial(1, c3_i21_p2) == 1 and p2_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_third_promo()
                day.add_customer_data(customer_data)
                p2_num -= p2_num
            elif np.random.binomial(1, c3_i21_p3) == 1 and p3_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_fourth_promo()
                day.add_customer_data(customer_data)
                p3_num -= p3_num

        for customer in range(c4_daily):
            customer_data = CustomerData(customer+1, group4.get_number())
            if np.random.binomial(1, c4_i1) == 1:
                promo = np.random.randint(1, 4)  # We can give it a promo (P1, P2, P3)
                give_promo = np.random.binomial()  # With a 50% chance it gets P0 and with a 50% chance it gets P1, P2 or P3 according to the previous result
                if give_promo == 0 and p0_num > 0:  # It gets P0
                    customer_data.set_true_first_promo()
                    p0_num -= p0_num
                elif promo == 1 and p1_num > 0:  # It gets P1
                    customer_data.set_true_second_promo()
                    p1_num -= p1_num
                elif promo == 2 and p2_num > 0:  # It gets P2
                    customer_data.set_true_third_promo()
                    p2_num -= p2_num
                elif promo == 3 and p3_num > 0:  # It gets P3
                    customer_data.set_true_fourth_promo()
                    p3_num -= p3_num
                customer_data.set_true_first_purchase()
                day.add_customer_data(customer_data)
            elif np.random.binomial(1, c4_i2) == 1:
                customer_data.set_true_second_purchase()
                day.add_customer_data(customer_data)
            elif np.random.binomial(1, c4_i21_p0) == 1 and p0_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_first_promo()
                day.add_customer_data(customer_data)
                p0_num -= p0_num
            elif np.random.binomial(1, c4_i21_p1) == 1 and p1_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_second_promo()
                day.add_customer_data(customer_data)
                p1_num -= p1_num
            elif np.random.binomial(1, c4_i21_p2) == 1 and p2_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_third_promo()
                day.add_customer_data(customer_data)
                p2_num -= p2_num
            elif np.random.binomial(1, c4_i21_p3) == 1 and p3_num > 0:
                customer_data.set_true_first_purchase()
                customer_data.set_true_second_purchase()
                customer_data.set_true_fourth_promo()
                day.add_customer_data(customer_data)
                p3_num -= p3_num
