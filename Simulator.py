from sklearn.preprocessing import normalize
from Day import *
from Group import *
from CustomerData import *
from Data import *
from Item import *
from LP_optimization import *

import numpy as np
from numpy.random import normal, binomial, uniform, choice

np.random.seed(1234)


class Simulator:
    def __init__(self, num_days):
        # Pay attention that in other simulations we need to overwrite group data
        self.customers_groups = np.array([Group(1), Group(2), Group(3), Group(4)])
        self.item1 = Item("Apple Watch", 300, 88)
        self.item2 = Item("Personalized wristband", 50, 16)
        self.days = []  # Old : Day(1, self.group1, self.group2, self.group3, self.group4)
        self.num_days = num_days
        for i in range(self.num_days):
            self.days.append(Day(i + 1, self.customers_groups[0], self.customers_groups[1], self.customers_groups[2],
                                 self.customers_groups[3]))
        self.daily_promos = np.zeros(4)
        self.daily_promos_temp = np.zeros(4)
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
            daily_customers = np.array([int(normal(data.get_n(1), 12)),
                                        int(normal(data.get_n(2), 14)),
                                        int(normal(data.get_n(3), 16)),
                                        int(normal(data.get_n(4), 17))])

            # Number of promo codes available (fixed fraction of the daily number of customers)
            self.daily_promos = [int(sum(daily_customers) * p0_frac),
                                 int(sum(daily_customers) * p1_frac),
                                 int(sum(daily_customers) * p2_frac),
                                 int(sum(daily_customers) * p3_frac)]
            # We create a copy to decrement the value in customer_purchase function since the original must not be
            # modified since it will be used in the optimization algorithm
            self.daily_promos_temp = self.daily_promos.copy()

            # Probability that a customer of a class buys the first item (in general - NOT "ONLY" ITEM 1) = Binomial
            prob_buy_item1 = np.array([binomial(daily_customers[0], data.get_i1_param(1)) / daily_customers[0],
                                       binomial(daily_customers[1], data.get_i1_param(2)) / daily_customers[1],
                                       binomial(daily_customers[2], data.get_i1_param(3)) / daily_customers[2],
                                       binomial(daily_customers[3], data.get_i1_param(4)) / daily_customers[3]])

            # Probability that a customer of a class buys the second item alone = Binomial TODO: values by hand are ok?
            prob_buy_item2 = np.array([binomial(daily_customers[0], 0.276) / daily_customers[0],
                                       binomial(daily_customers[1], 0.421) / daily_customers[1],
                                       binomial(daily_customers[2], 0.358) / daily_customers[2],
                                       binomial(daily_customers[3], 0.452) / daily_customers[3]])

            # Probability that a customer of a class buys the second item given the first + each promo
            # The matrix structure is the usual one (rows: promo code; column: customer group)
            prob_buy_item21 = np.array([  # Promo code P0
                                        [binomial(daily_customers[0], data.get_i21_p0_param(1)) / daily_customers[0],
                                         binomial(daily_customers[1], data.get_i21_p0_param(2)) / daily_customers[1],
                                         binomial(daily_customers[2], data.get_i21_p0_param(3)) / daily_customers[2],
                                         binomial(daily_customers[3], data.get_i21_p0_param(4)) / daily_customers[3]],
                                        # Promo code P1
                                        [binomial(daily_customers[0], data.get_i21_p1_param(1)) / daily_customers[0],
                                         binomial(daily_customers[1], data.get_i21_p1_param(2)) / daily_customers[1],
                                         binomial(daily_customers[2], data.get_i21_p1_param(3)) / daily_customers[2],
                                         binomial(daily_customers[3], data.get_i21_p1_param(4)) / daily_customers[3]],
                                        # Promo code P2
                                        [binomial(daily_customers[0], data.get_i21_p2_param(1)) / daily_customers[0],
                                         binomial(daily_customers[1], data.get_i21_p2_param(2)) / daily_customers[1],
                                         binomial(daily_customers[2], data.get_i21_p2_param(3)) / daily_customers[2],
                                         binomial(daily_customers[3], data.get_i21_p2_param(4)) / daily_customers[3]],
                                        # Promo code P3
                                        [binomial(daily_customers[0], data.get_i21_p3_param(1)) / daily_customers[0],
                                         binomial(daily_customers[1], data.get_i21_p3_param(2)) / daily_customers[1],
                                         binomial(daily_customers[2], data.get_i21_p3_param(3)) / daily_customers[2],
                                         binomial(daily_customers[3], data.get_i21_p3_param(4)) / daily_customers[3]]
                                    ])

            # For each customer, we extract its group using the function np.random.choice().
            # This allows us to extract groups keeping the same proportions as the daily number of customers per group.
            # Then, we create the CustomerData object accordingly, and we draw from a Bernoulli for each kind of
            # possible purchase (buy item 1 in general, buy only item 2, buy item 2 after buying item 1 with promos).
            # Finally, we call the customer_purchase_step1 function passing the extracted numbers, so that the
            # attributes of the CustomerData object are set accordingly. Also, every time we give a promo to the
            # customer, we decrease the number of promo codes available for that specific promo code.
            for customer in range(sum(daily_customers)):
                group = choice(4, p=[daily_customers[0] / sum(daily_customers),
                                     daily_customers[1] / sum(daily_customers),
                                     daily_customers[2] / sum(daily_customers),
                                     daily_customers[3] / sum(daily_customers)])

                customer_data = CustomerData(customer + 1, self.customers_groups[group].get_number())
                selected_promo = int(uniform(0, 4))
                self.customer_purchase(day, customer_data, selected_promo, prob_buy_item1[group], prob_buy_item2[group],
                                       prob_buy_item21[0][group], prob_buy_item21[1][group], prob_buy_item21[2][group],
                                       prob_buy_item21[3][group])

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
                  self.daily_promos[0], self.daily_promos[1], self.daily_promos[2], self.daily_promos[3],
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
            daily_customers = np.array([int(normal(data.get_n(1), 12)),
                                        int(normal(data.get_n(2), 14)),
                                        int(normal(data.get_n(3), 16)),
                                        int(normal(data.get_n(4), 17))])

            # Number of promo codes available (fixed fraction of the daily number of customers)
            self.daily_promos = [int(sum(daily_customers) * p0_frac),
                                 int(sum(daily_customers) * p1_frac),
                                 int(sum(daily_customers) * p2_frac),
                                 int(sum(daily_customers) * p3_frac)]
            # We create a copy to decrement the value in customer_purchase function since the original must not be
            # modified since it will be used in the optimization algorithm
            self.daily_promos_temp = self.daily_promos.copy()

            # Probability that a customer of a class buys the first item (in general - NOT "ONLY" ITEM 1) = Binomial
            prob_buy_item1 = np.array([binomial(daily_customers[0], data.get_i1_param(1)) / daily_customers[0],
                                       binomial(daily_customers[1], data.get_i1_param(2)) / daily_customers[1],
                                       binomial(daily_customers[2], data.get_i1_param(3)) / daily_customers[2],
                                       binomial(daily_customers[3], data.get_i1_param(4)) / daily_customers[3]])

            # Probability that a customer of a class buys the second item alone = Binomial TODO: values by hand are ok?
            prob_buy_item2 = np.array([binomial(daily_customers[0], 0.276) / daily_customers[0],
                                       binomial(daily_customers[1], 0.421) / daily_customers[1],
                                       binomial(daily_customers[2], 0.358) / daily_customers[2],
                                       binomial(daily_customers[3], 0.452) / daily_customers[3]])

            # Probability that a customer of a class buys the second item given the first + each promo
            # The matrix structure is the usual one (rows: promo code; column: customer group)
            prob_buy_item21 = np.array([  # Promo code P0
                                        [binomial(daily_customers[0], data.get_i21_p0_param(1)) / daily_customers[0],
                                         binomial(daily_customers[1], data.get_i21_p0_param(2)) / daily_customers[1],
                                         binomial(daily_customers[2], data.get_i21_p0_param(3)) / daily_customers[2],
                                         binomial(daily_customers[3], data.get_i21_p0_param(4)) / daily_customers[3]],
                                        # Promo code P1
                                        [binomial(daily_customers[0], data.get_i21_p1_param(1)) / daily_customers[0],
                                         binomial(daily_customers[1], data.get_i21_p1_param(2)) / daily_customers[1],
                                         binomial(daily_customers[2], data.get_i21_p1_param(3)) / daily_customers[2],
                                         binomial(daily_customers[3], data.get_i21_p1_param(4)) / daily_customers[3]],
                                        # Promo code P2
                                        [binomial(daily_customers[0], data.get_i21_p2_param(1)) / daily_customers[0],
                                         binomial(daily_customers[1], data.get_i21_p2_param(2)) / daily_customers[1],
                                         binomial(daily_customers[2], data.get_i21_p2_param(3)) / daily_customers[2],
                                         binomial(daily_customers[3], data.get_i21_p2_param(4)) / daily_customers[3]],
                                        # Promo code P3
                                        [binomial(daily_customers[0], data.get_i21_p3_param(1)) / daily_customers[0],
                                         binomial(daily_customers[1], data.get_i21_p3_param(2)) / daily_customers[1],
                                         binomial(daily_customers[2], data.get_i21_p3_param(3)) / daily_customers[2],
                                         binomial(daily_customers[3], data.get_i21_p3_param(4)) / daily_customers[3]]
                                    ])

            # For each customer, we extract its group using the function np.random.choice().
            # This allows us to extract groups keeping the same proportions as the daily number of customers per group.
            # Then, we create the CustomerData object accordingly, and we draw from a Bernoulli for each kind of
            # possible purchase (buy item 1 in general, buy only item 2, buy item 2 after buying item 1 with promos).
            # Finally, we call the customer_purchase_step1 function passing the extracted numbers, so that the
            # attributes of the CustomerData object are set accordingly. Also, every time we give a promo to the
            # customer, we decrease the number of promo codes available for that specific promo code.
            # The main difference with respect to the previous step is that here the promo that we give to the
            # customer that bought item 1 is not drawn uniformly (except for the first day) but using the result
            # of the optimization algorithm.
            for customer in range(sum(daily_customers)):
                group = choice(4, p=[daily_customers[0] / sum(daily_customers),
                                     daily_customers[1] / sum(daily_customers),
                                     daily_customers[2] / sum(daily_customers),
                                     daily_customers[3] / sum(daily_customers)])

                customer_data = CustomerData(customer + 1, self.customers_groups[group].get_number())
                selected_promo = choice(4, p=[prob_promo[0][group], prob_promo[1][group], prob_promo[2][group],
                                              prob_promo[3][group]])
                self.customer_purchase(day, customer_data, selected_promo, prob_buy_item1[group], prob_buy_item2[group],
                                       prob_buy_item21[0][group], prob_buy_item21[1][group], prob_buy_item21[2][group],
                                       prob_buy_item21[3][group])

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

            # We run the linear optimization algorithm at the end of the day. It uses the data up to this day
            update = LP(self.item2.get_price(), self.discount_p1, self.discount_p2, self.discount_p3,
                        avg_conversion_rates[0][0], avg_conversion_rates[0][1], avg_conversion_rates[0][2], avg_conversion_rates[0][3],
                        avg_conversion_rates[1][0], avg_conversion_rates[1][1], avg_conversion_rates[1][2], avg_conversion_rates[1][3],
                        avg_conversion_rates[2][0], avg_conversion_rates[2][1], avg_conversion_rates[2][2], avg_conversion_rates[2][3],
                        avg_conversion_rates[3][0], avg_conversion_rates[3][1], avg_conversion_rates[3][2], avg_conversion_rates[3][3],
                        self.daily_promos[0], self.daily_promos[1], self.daily_promos[2], self.daily_promos[3], # Maybe count also these? #TODO
                        avg_num_customers[0], avg_num_customers[1], avg_num_customers[2], avg_num_customers[3])

            lp_matrix = update[1]
            print(lp_matrix)  # Just for visualizing the daily optimization results TODO print once a month maybe?
            prob_promo = normalize(lp_matrix, 'l1', axis=0)

########################################################################################################################

    def customer_purchase(self, day, customer_data, selected_promo, buy1, buy2, buy21_p0, buy21_p1, buy21_p2, buy21_p3):
        if binomial(1, buy1) == 1:
            customer_data.buy_item1()
            if selected_promo == 0 and self.daily_promos_temp[0] > 0:
                customer_data.give_p0()
                self.daily_promos_temp[0] -= 1
                if binomial(1, buy21_p0) == 1:
                    customer_data.buy_item2()
            elif selected_promo == 1 and self.daily_promos_temp[1] > 0:
                customer_data.give_p1()
                self.daily_promos_temp[1] -= 1
                if binomial(1, buy21_p1) == 1:
                    customer_data.buy_item2()
            elif selected_promo == 2 and self.daily_promos_temp[2] > 0:
                customer_data.give_p2()
                self.daily_promos_temp[2] -= 1
                if binomial(1, buy21_p2) == 1:
                    customer_data.buy_item2()
            elif selected_promo == 3 and self.daily_promos_temp[3] > 0:
                customer_data.give_p3()
                self.daily_promos_temp[3] -= 1
                if binomial(1, buy21_p3) == 1:
                    customer_data.buy_item2()
            day.add_customer_data(customer_data)  # We add the customer only if he bought something
        elif binomial(1, buy2) == 1:
            customer_data.buy_item2()
            day.add_customer_data(customer_data)  # We add the customer only if he bought something

########################################################################################################################

    def get_days(self):
        return self.days

    def get_item1(self):
        return self.item1

    def get_item2(self):
        return self.item2

    def get_daily_promos(self):
        return self.daily_promos
