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
        self.p0_num = 0
        self.p1_num = 0
        self.p2_num = 0
        self.p3_num = 0

    def simulation_step_1(self):
        # Creating the Data object to get the actual numbers from the Google Module
        data = Data()

        # Fixing the fractions of promo codes available
        p0_frac = 0.7
        p1_frac = 0.2
        p2_frac = 0.07
        p3_frac = 0.03

        # Number of customers per class = Gaussian TODO: are sigmas correct?
        c1_daily = int(np.random.normal(data.get_n(1), 12))
        c2_daily = int(np.random.normal(data.get_n(2), 14))
        c3_daily = int(np.random.normal(data.get_n(3), 16))
        c4_daily = int(np.random.normal(data.get_n(4), 17))

        # Number of promo codes available
        self.p0_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p0_frac)
        self.p1_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p1_frac)
        self.p2_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p2_frac)
        self.p3_num = int((c1_daily + c2_daily + c3_daily + c4_daily) * p3_frac)

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

        # For each class of customers, we create a CustomerData object.
        # Then, we draw from a Binomial for each kind of purchase possible, using the numbers computed above.
        # We call the function customer_purchase passing the extracted numbers, so that the attributes of the
        # CustomerData object are set accordingly.
        # Every time we give a promo to the customer, we decrease the number of coupons available for that promo code

        # Customer class 1
        for customer in range(c1_daily):
            customer_data = CustomerData(customer+1, self.group1.get_number())
            buy1 = np.random.binomial(1, c1_i1)
            buy2 = np.random.binomial(1, c1_i2)
            buy21_p0 = np.random.binomial(1, c1_i21_p0)
            buy21_p1 = np.random.binomial(1, c1_i21_p1)
            buy21_p2 = np.random.binomial(1, c1_i21_p2)
            buy21_p3 = np.random.binomial(1, c1_i21_p3)
            self.customer_purchase(customer_data, buy1, buy2, buy21_p0, buy21_p1, buy21_p2, buy21_p3)

        # Customer class 2
        for customer in range(c2_daily):
            customer_data = CustomerData(customer+1, self.group2.get_number())
            buy1 = np.random.binomial(1, c2_i1)
            buy2 = np.random.binomial(1, c2_i2)
            buy21_p0 = np.random.binomial(1, c2_i21_p0)
            buy21_p1 = np.random.binomial(1, c2_i21_p1)
            buy21_p2 = np.random.binomial(1, c2_i21_p2)
            buy21_p3 = np.random.binomial(1, c2_i21_p3)
            self.customer_purchase(customer_data, buy1, buy2, buy21_p0, buy21_p1, buy21_p2, buy21_p3)

        # Customer class 3
        for customer in range(c3_daily):
            customer_data = CustomerData(customer+1, self.group3.get_number())
            buy1 = np.random.binomial(1, c3_i1)
            buy2 = np.random.binomial(1, c3_i2)
            buy21_p0 = np.random.binomial(1, c3_i21_p0)
            buy21_p1 = np.random.binomial(1, c3_i21_p1)
            buy21_p2 = np.random.binomial(1, c3_i21_p2)
            buy21_p3 = np.random.binomial(1, c3_i21_p3)
            self.customer_purchase(customer_data, buy1, buy2, buy21_p0, buy21_p1, buy21_p2, buy21_p3)

        # Customer class 4
        for customer in range(c4_daily):
            customer_data = CustomerData(customer+1, self.group4.get_number())
            buy1 = np.random.binomial(1, c4_i1)
            buy2 = np.random.binomial(1, c4_i2)
            buy21_p0 = np.random.binomial(1, c4_i21_p0)
            buy21_p1 = np.random.binomial(1, c4_i21_p1)
            buy21_p2 = np.random.binomial(1, c4_i21_p2)
            buy21_p3 = np.random.binomial(1, c4_i21_p3)
            self.customer_purchase(customer_data, buy1, buy2, buy21_p0, buy21_p1, buy21_p2, buy21_p3)

    def customer_purchase(self, customer_data, buy1, buy2, buy21_p0, buy21_p1, buy21_p2, buy21_p3):
        if buy1 == 1:
            # We extract uniformly between promos P1, P2 and P3
            promo = np.random.randint(1, 4)
            # And we draw from a Binomial to decide if the customer gets P0 or the previous extracted promo
            give_promo = np.random.binomial()
            if give_promo == 0 and self.p0_num > 0:             # It gets P0
                customer_data.set_true_first_promo()
                self.p0_num -= 1
            elif promo == 1 and self.p1_num > 0:                # It gets P1
                customer_data.set_true_second_promo()
                self.p1_num -= 1
            elif promo == 2 and self.p2_num > 0:                # It gets P2
                customer_data.set_true_third_promo()
                self.p2_num -= 1
            elif promo == 3 and self.p3_num > 0:                # It gets P3
                customer_data.set_true_fourth_promo()
                self.p3_num -= 1
            customer_data.set_true_first_purchase()
            self.day.add_customer_data(customer_data)
        # Else, if the customer buys item 2 alone
        elif buy2 == 1:
            customer_data.set_true_second_purchase()
            self.day.add_customer_data(customer_data)
        # Else, if the customer buys item 2 after buying item 1 and having P0
        elif buy21_p0 == 1 and self.p0_num > 0:
            customer_data.set_true_first_purchase()
            customer_data.set_true_second_purchase()
            customer_data.set_true_first_promo()
            self.day.add_customer_data(customer_data)
            self.p0_num -= 1
        # Else, if the customer buys item 2 after buying item 1 and having P1
        elif buy21_p1 and self.p1_num > 0:
            customer_data.set_true_first_purchase()
            customer_data.set_true_second_purchase()
            customer_data.set_true_second_promo()
            self.day.add_customer_data(customer_data)
            self.p1_num -= 1
        # Else, if the customer buys item 2 after buying item 1 and having P2
        elif buy21_p2 and self.p2_num > 0:
            customer_data.set_true_first_purchase()
            customer_data.set_true_second_purchase()
            customer_data.set_true_third_promo()
            self.day.add_customer_data(customer_data)
            self.p2_num -= 1
        # Else, if the customer buys item 2 after buying item 1 and having P3
        elif buy21_p3 and self.p3_num > 0:
            customer_data.set_true_first_purchase()
            customer_data.set_true_second_purchase()
            customer_data.set_true_fourth_promo()
            self.day.add_customer_data(customer_data)
            self.p3_num -= 1

    def get_day_step1(self):
        return self.day
