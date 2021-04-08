from sklearn.preprocessing import normalize
from Day import *
from Group import *
from CustomerData import *
from Data import *
from Item import *
from LP_optimization import *

import numpy as np
np.random.seed(1234)


class Simulator:
    def __init__(self, num_days):
        # Pay attention that in other simulations we need to overwrite group data
        self.group1 = Group(1)
        self.group2 = Group(2)
        self.group3 = Group(3)
        self.group4 = Group(4)
        self.item1 = Item("Apple Watch", 300, 88)
        self.item2 = Item("Personalized wristband", 50, 16)
        self.days = []                                # Old : Day(1, self.group1, self.group2, self.group3, self.group4)
        self.num_days = num_days
        for i in range(self.num_days):
            self.days.append(Day(i+1, self.group1, self.group2, self.group3, self.group4))
        self.p0_num = 0
        self.p1_num = 0
        self.p2_num = 0
        self.p3_num = 0
        self.p0_temp = 0
        self.p1_temp = 0
        self.p2_temp = 0
        self.p3_temp = 0
        self.discount_p1 = 0.1
        self.discount_p2 = 0.2
        self.discount_p3 = 0.5

    def simulation_step_1(self, p0_frac, p1_frac, p2_frac, p3_frac):
        # Creating the Data object to get the actual numbers from the Google Module
        data = Data()

        # Useful structures to compute the conversion rates and customers number averages at the end
        avg_conversion_rates = np.zeros((4, 4))
        avg_num_customers = np.zeros(4)

        # For every day, we generate all the customer data according to probability distributions
        # Then, we update the conversion rates for that day
        for day in self.days:

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
            self.p0_temp = self.p0_num
            self.p1_temp = self.p1_num
            self.p2_temp = self.p2_num
            self.p3_temp = self.p3_num

            # Probability that a customer of a class buys the first item (in general - NOT "ONLY" ITEM 1) = Binomial
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

            # For each customer, we extract its group using the function np.random.choice().
            # This allows us to extract groups keeping the same proportions as the daily number of customers per group.
            # Then, we create the CustomerData object accordingly, and we draw from a Bernoulli for each kind of
            # possible purchase (buy item 1 in general, buy only item 2, buy item 2 after buying item 1 with promos).
            # Finally, we call the customer_purchase_step1 function passing the extracted numbers, so that the
            # attributes of the CustomerData object are set accordingly. Also, every time we give a promo to the
            # customer, we decrease the number of promo codes available for that specific promo code.

            total_number_of_customers = c1_daily + c2_daily + c3_daily + c4_daily
            for customer in range(total_number_of_customers):
                group = np.random.choice(np.arange(1, 5), p=[c1_daily / total_number_of_customers,
                                                             c2_daily / total_number_of_customers,
                                                             c3_daily / total_number_of_customers,
                                                             c4_daily / total_number_of_customers])
                if group == 1:
                    customer_data = CustomerData(customer + 1, self.group1.get_number())
                    promo = int(np.random.uniform(0, 4))
                    self.customer_purchase(day, customer_data, promo, c1_i1, c1_i2, c1_i21_p0, c1_i21_p1, c1_i21_p2, c1_i21_p3)
                elif group == 2:
                    customer_data = CustomerData(customer + 1, self.group2.get_number())
                    promo = int(np.random.uniform(0, 4))
                    self.customer_purchase(day, customer_data, promo, c2_i1, c2_i2, c2_i21_p0, c2_i21_p1, c2_i21_p2, c2_i21_p3)
                elif group == 3:
                    customer_data = CustomerData(customer + 1, self.group3.get_number())
                    promo = int(np.random.uniform(0, 4))
                    self.customer_purchase(day, customer_data, promo, c3_i1, c3_i2, c3_i21_p0, c3_i21_p1, c3_i21_p2, c3_i21_p3)
                elif group == 4:
                    customer_data = CustomerData(customer + 1, self.group4.get_number())
                    promo = int(np.random.uniform(0, 4))
                    self.customer_purchase(day, customer_data, promo, c4_i1, c4_i2, c4_i21_p0, c4_i21_p1, c4_i21_p2, c4_i21_p3)

            # After the CustomerData objects for the day are created, we compute all the conversion rates for the day
            day.set_conversion_rates()

            # We add the daily conversion rates to the matrix in Day (useful to compute the average later)
            avg_conversion_rates += day.get_conversion_rates_item_21()
            # We add the daily number of customers to the matrix in Day (useful to compute the average later)
            avg_num_customers += day.get_customers_purchases()

        # After having initialized the data for all the days, we compute the average conversion rate
        avg_conversion_rates = avg_conversion_rates / self.num_days
        # And the average number of customers
        avg_num_customers = avg_num_customers / self.num_days

        # Finally, we call the linear optimization algorithm
        return LP(self.item2.get_price(), self.discount_p1, self.discount_p2, self.discount_p3,
                  avg_conversion_rates[0][0], avg_conversion_rates[0][1], avg_conversion_rates[0][2], avg_conversion_rates[0][3],
                  avg_conversion_rates[1][0], avg_conversion_rates[1][1], avg_conversion_rates[1][2], avg_conversion_rates[1][3],
                  avg_conversion_rates[2][0], avg_conversion_rates[2][1], avg_conversion_rates[2][2], avg_conversion_rates[2][3],
                  avg_conversion_rates[3][0], avg_conversion_rates[3][1], avg_conversion_rates[3][2], avg_conversion_rates[3][3],
                  self.p0_num, self.p1_num, self.p2_num, self.p3_num,
                  avg_num_customers[0], avg_num_customers[1], avg_num_customers[2], avg_num_customers[3])

########################################################################################################################

    def simulation_step_2(self, p0_frac, p1_frac, p2_frac, p3_frac):
        # Creating the Data object to get the actual numbers from the Google Module
        data = Data()

        # Useful structures to compute the conversion rates and customers number averages at the end
        sum_conversion_rates = np.zeros((4, 4))
        sum_num_customers = np.zeros(4)

        # Matrix which contains the probabilities to give each promo code to every customer group
        # The structure is the usual (rows: promo codes; columns: customer groups)
        # Initially we give a promo code with uniform probability to each class (sum of each column = 1)
        # Then, after each day, this matrix will be updated according to the result of the optimization algorithm
        prob_promo = np.full((4, 4), 0.25)

        # For every day, we generate all the customer data according to probability distributions.
        # Then, we update the conversion rates for that day and we compute the average of the conversion rates
        # and of the number of customers up to that day.
        # Finally, the optimization algorithm is run using these averages, and the prob_promo matrix is updated.
        for day in self.days:

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
            self.p0_temp = self.p0_num
            self.p1_temp = self.p1_num
            self.p2_temp = self.p2_num
            self.p3_temp = self.p3_num

            # Probability that a customer of a class buys the first item (in general - NOT "ONLY" ITEM 1) = Binomial
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

            # For each customer, we extract its group using the function np.random.choice().
            # This allows us to extract groups keeping the same proportions as the daily number of customers per group.
            # Then, we create the CustomerData object accordingly, and we draw from a Bernoulli for each kind of
            # possible purchase (buy item 1 in general, buy only item 2, buy item 2 after buying item 1 with promos).
            # Finally, we call the customer_purchase_step1 function passing the extracted numbers, so that the
            # attributes of the CustomerData object are set accordingly. Also, every time we give a promo to the
            # customer, we decrease the number of promo codes available for that specific promo code.
            # The main different with respect to the previous step is that here the promo that we give to the
            # customer that bought item 1 is not drawn uniformly (except for the first day) but using the result
            # of the optimization algorithm.

            total_number_of_customers = c1_daily + c2_daily + c3_daily + c4_daily
            for customer in range(total_number_of_customers):
                group = np.random.choice(np.arange(1, 5), p=[c1_daily / total_number_of_customers,
                                                             c2_daily / total_number_of_customers,
                                                             c3_daily / total_number_of_customers,
                                                             c4_daily / total_number_of_customers])
                if group == 1:
                    customer_data = CustomerData(customer + 1, self.group1.get_number())
                    promo = np.random.choice(4, p=[prob_promo[0][0], prob_promo[1][0], prob_promo[2][0], prob_promo[3][0]])
                    self.customer_purchase(day, customer_data, promo, c1_i1, c1_i2, c1_i21_p0, c1_i21_p1, c1_i21_p2, c1_i21_p3)
                elif group == 2:
                    customer_data = CustomerData(customer + 1, self.group2.get_number())
                    promo = np.random.choice(4, p=[prob_promo[0][1], prob_promo[1][1], prob_promo[2][1], prob_promo[3][1]])
                    self.customer_purchase(day, customer_data, promo, c2_i1, c2_i2, c2_i21_p0, c2_i21_p1, c2_i21_p2, c2_i21_p3)
                elif group == 3:
                    customer_data = CustomerData(customer + 1, self.group3.get_number())
                    promo = np.random.choice(4, p=[prob_promo[0][2], prob_promo[1][2], prob_promo[2][2], prob_promo[3][2]])
                    self.customer_purchase(day, customer_data, promo, c3_i1, c3_i2, c3_i21_p0, c3_i21_p1, c3_i21_p2, c3_i21_p3)
                elif group == 4:
                    customer_data = CustomerData(customer + 1, self.group4.get_number())
                    promo = np.random.choice(4, p=[prob_promo[0][3], prob_promo[1][3], prob_promo[2][3], prob_promo[3][3]])
                    self.customer_purchase(day, customer_data, promo, c4_i1, c4_i2, c4_i21_p0, c4_i21_p1, c4_i21_p2, c4_i21_p3)

            # After the CustomerData objects for the day are created, we compute all the conversion rates for the day
            day.set_conversion_rates()

            # We compute the average of the conversion rates up to this day
            # Notice that "sum_conversion_rates" is defined before the loop "for day in self.days"
            sum_conversion_rates += day.get_conversion_rates_item_21()
            avg_conversion_rates = sum_conversion_rates / day.get_id()

            # We compute the average of the number of customers up to this day
            # Notice that "sum_num_customers" is defined before the loop "for day in self.days"
            sum_num_customers += day.get_customers_purchases()
            avg_num_customers = sum_num_customers / day.get_id()

            # We call the linear optimization algorithm at the end of the day. It uses the data up to this day
            update = LP(self.item2.get_price(), self.discount_p1, self.discount_p2, self.discount_p3,
                        avg_conversion_rates[0][0], avg_conversion_rates[0][1], avg_conversion_rates[0][2], avg_conversion_rates[0][3],
                        avg_conversion_rates[1][0], avg_conversion_rates[1][1], avg_conversion_rates[1][2], avg_conversion_rates[1][3],
                        avg_conversion_rates[2][0], avg_conversion_rates[2][1], avg_conversion_rates[2][2], avg_conversion_rates[2][3],
                        avg_conversion_rates[3][0], avg_conversion_rates[3][1], avg_conversion_rates[3][2], avg_conversion_rates[3][3],
                        self.p0_num, self.p1_num, self.p2_num, self.p3_num,         # Maybe count also these? #TODO
                        avg_num_customers[0], avg_num_customers[1], avg_num_customers[2], avg_num_customers[3])

            lp_matrix = update[1]
            print(lp_matrix)                # Just for visualizing the daily optimization results TODO change at the end
            prob_promo = normalize(lp_matrix, 'l1', axis=0)

            #prob_promo[prob_promo == 0] = 0.001                 # Adding some noise since we don't want zero probability
            #prob_promo = normalize(prob_promo, 'l1', axis=0)

########################################################################################################################

    def customer_purchase(self, day, customer_data, promo, buy1, buy2, buy21_p0, buy21_p1, buy21_p2, buy21_p3):
        if np.random.binomial(1, buy1) == 1:
            customer_data.buy_item1()
            if promo == 0 and self.p0_temp > 0:
                customer_data.give_p0()
                self.p0_temp -= 1
                if np.random.binomial(1, buy21_p0) == 1:
                    customer_data.buy_item2()
            elif promo == 1 and self.p1_temp > 0:
                customer_data.give_p1()
                self.p1_temp -= 1
                if np.random.binomial(1, buy21_p1) == 1:
                    customer_data.buy_item2()
            elif promo == 2 and self.p2_temp > 0:
                customer_data.give_p2()
                self.p2_temp -= 1
                if np.random.binomial(1, buy21_p2) == 1:
                    customer_data.buy_item2()
            elif promo == 3 and self.p3_temp > 0:
                customer_data.give_p3()
                self.p3_temp -= 1
                if np.random.binomial(1, buy21_p3) == 1:
                    customer_data.buy_item2()
            day.add_customer_data(customer_data)                       # We add the customer only if he bought something
        elif np.random.binomial(1, buy2) == 1:
            customer_data.buy_item2()
            day.add_customer_data(customer_data)                       # We add the customer only if he bought something

########################################################################################################################

    def get_day_step1(self):
        return self.days

    def get_item1(self):
        return self.item1

    def get_item2(self):
        return self.item2

    def get_p0_num(self):
        return self.p0_num

    def get_p1_num(self):
        return self.p1_num

    def get_p2_num(self):
        return self.p2_num

    def get_p3_num(self):
        return self.p3_num
