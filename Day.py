import numpy as np


class Day:
    def __init__(self, identifier, group1, group2, group3, group4):
        self.identifier = identifier                # Identifier of the day
        self.customers_data_list = []               # List of CustomerData objects TODO: useless?
        self.group1 = group1                        # Reference to Group 1
        self.group2 = group2                        # Reference to Group 2
        self.group3 = group3                        # Reference to Group 3
        self.group4 = group4                        # Reference to Group 4

        # Statistics of the single day
        # Do not care about people who do not buy anything since the aim is to maximize purchases of item2 given item1
        # The arrays/matrices structure is always the same (rows: promo, columns: customer group)
        # The row index corresponds exactly to the promo code (0: P0, 1: P1, 2: P2, 3: P3)
        # The column index corresponds to the customer group minus one (0: group1, 1: group2, 2: group3, 3: group4)

        self.customers_purchases = np.zeros(4)      # Number of customers who bought something (per group)

        self.purchases_item2 = np.zeros(4)          # Number of customers purchasing item2 in general (per group)
        self.purchases_item1 = np.zeros((4, 4))     # Number of customers purchasing item 1 (and getting a promo)
        self.purchases_item21 = np.zeros((4, 4))    # Number of customers purchasing item 2 given item 1 (and promo)

        self.conversion_rates_item1 = np.zeros(4)          # Conversion rates for buying item 1
        self.conversion_rates_item21 = np.zeros((4, 4))    # Conversion rates for buying item 2 given item 1 (and promo)

    def get_id(self):
        return self.identifier

    def get_customers_purchases(self):
        return self.customers_purchases

    def get_customers_data_list(self):
        return self.customers_data_list

    def get_conversion_rates_item1(self):
        return self.conversion_rates_item1

    def get_conversion_rates_item_21(self):
        return self.conversion_rates_item21

    def add_customer_data(self, customer_data):
        self.customers_data_list.append(customer_data)          # Adding the CustomerData object to the list
        customer_group = customer_data.get_group() - 1          # Saving the customer group (-1 to be used as index)
        # If the customer bought the second item (in general)
        if customer_data.purchased_item2():
            self.purchases_item2[customer_group] += 1
        # If the customer bought the first item
        if customer_data.purchased_item1():
            self.customers_purchases[customer_group] += 1
            # According to the promo obtained and the fact that the customer bought also the second item or not,
            # we increase the corresponding element in the purchases counters of the day
            if customer_data.received_p0():
                if customer_data.purchased_item2():
                    self.purchases_item21[0][customer_group] += 1
                else:
                    self.purchases_item1[0][customer_group] += 1
            elif customer_data.received_p1():
                if customer_data.purchased_item2():
                    self.purchases_item21[1][customer_group] += 1
                else:
                    self.purchases_item1[1][customer_group] += 1
            elif customer_data.received_p2():
                if customer_data.purchased_item2():
                    self.purchases_item21[2][customer_group] += 1
                else:
                    self.purchases_item1[2][customer_group] += 1
            elif customer_data.received_p3():
                if customer_data.purchased_item2():
                    self.purchases_item21[3][customer_group] += 1
                else:
                    self.purchases_item1[3][customer_group] += 1

    def set_conversion_rates(self):
        # purchase1 per class / total_purchase per class
        if self.customers_purchases[0] > 0:
            self.conversion_rates_item1[0] = (self.purchases_item1[0][0] + self.purchases_item1[1][0] + self.purchases_item1[2][0] + self.purchases_item1[3][0] + self.purchases_item21[0][0] + self.purchases_item21[1][0] + self.purchases_item21[2][0] + self.purchases_item21[3][0])/self.customers_purchases[0]
        if self.customers_purchases[1] > 0:
            self.conversion_rates_item1[1] = (self.purchases_item1[0][1] + self.purchases_item1[1][1] + self.purchases_item1[2][1] + self.purchases_item1[3][1] + self.purchases_item21[0][1] + self.purchases_item21[1][1] + self.purchases_item21[2][1] + self.purchases_item21[3][1])/self.customers_purchases[1]
        if self.customers_purchases[2] > 0:
            self.conversion_rates_item1[2] = (self.purchases_item1[0][2] + self.purchases_item1[1][2] + self.purchases_item1[2][2] + self.purchases_item1[3][2] + self.purchases_item21[0][2] + self.purchases_item21[1][2] + self.purchases_item21[2][2] + self.purchases_item21[3][2])/self.customers_purchases[2]
        if self.customers_purchases[3] > 0:
            self.conversion_rates_item1[3] = (self.purchases_item1[0][3] + self.purchases_item1[1][3] + self.purchases_item1[2][3] + self.purchases_item1[3][3] + self.purchases_item21[0][3] + self.purchases_item21[1][3] + self.purchases_item21[2][3] + self.purchases_item21[3][3])/self.customers_purchases[3]

        # purchase2 per class / total_purchase per class
        if self.customers_purchases[0] > 0:
            self.group1.set_conversion_rate_item2(self.purchases_item2[0] / self.customers_purchases[0])
        if self.customers_purchases[1] > 0:
            self.group2.set_conversion_rate_item2(self.purchases_item2[1] / self.customers_purchases[1])
        if self.customers_purchases[2] > 0:
            self.group2.set_conversion_rate_item2(self.purchases_item2[2] / self.customers_purchases[2])
        if self.customers_purchases[3] > 0:
            self.group2.set_conversion_rate_item2(self.purchases_item2[3] / self.customers_purchases[3])

        # purchase2 and purchase1 per class / purchase1 per class
        if (self.purchases_item1[0][0] + self.purchases_item21[0][0]) > 0:
            self.conversion_rates_item21[0][0] = self.purchases_item21[0][0] / (self.purchases_item1[0][0] + self.purchases_item21[0][0])
        if (self.purchases_item1[0][1] + self.purchases_item21[0][1]) > 0:
            self.conversion_rates_item21[0][1] = self.purchases_item21[0][1] / (self.purchases_item1[0][1] + self.purchases_item21[0][1])
        if (self.purchases_item1[0][2] + self.purchases_item21[0][2]) > 0:
            self.conversion_rates_item21[0][2] = self.purchases_item21[0][2] / (self.purchases_item1[0][2] + self.purchases_item21[0][2])
        if (self.purchases_item1[0][3] + self.purchases_item21[0][3]) > 0:
            self.conversion_rates_item21[0][3] = self.purchases_item21[0][3] / (self.purchases_item1[0][3] + self.purchases_item21[0][3])

        # purchase2 and purchase1 per class / purchase1 per class
        if (self.purchases_item1[1][0] + self.purchases_item21[1][0]) > 0:
            self.conversion_rates_item21[1][0] = self.purchases_item21[1][0] / (self.purchases_item1[1][0] + self.purchases_item21[1][0])
        if (self.purchases_item1[1][1] + self.purchases_item21[1][1]) > 0:
            self.conversion_rates_item21[1][1] = self.purchases_item21[1][1] / (self.purchases_item1[1][1] + self.purchases_item21[1][1])
        if (self.purchases_item1[1][2] + self.purchases_item21[1][2]) > 0:
            self.conversion_rates_item21[1][2] = self.purchases_item21[1][2] / (self.purchases_item1[1][2] + self.purchases_item21[1][2])
        if (self.purchases_item1[1][3] + self.purchases_item21[1][3]) > 0:
            self.conversion_rates_item21[1][3] = self.purchases_item21[1][3] / (self.purchases_item1[1][3] + self.purchases_item21[1][3])

        # purchase2 and purchase1 per class / purchase1 per class
        if (self.purchases_item1[2][0] + self.purchases_item21[2][0]) > 0:
            self.conversion_rates_item21[2][0] = self.purchases_item21[2][0] / (self.purchases_item1[2][0] + self.purchases_item21[2][0])
        if (self.purchases_item1[2][1] + self.purchases_item21[2][1]) > 0:
            self.conversion_rates_item21[2][1] = self.purchases_item21[2][1] / (self.purchases_item1[2][1] + self.purchases_item21[2][1])
        if (self.purchases_item1[2][2] + self.purchases_item21[2][2]) > 0:
            self.conversion_rates_item21[2][2] = self.purchases_item21[2][2] / (self.purchases_item1[2][2] + self.purchases_item21[2][2])
        if (self.purchases_item1[2][3] + self.purchases_item21[2][3]) > 0:
            self.conversion_rates_item21[2][3] = self.purchases_item21[2][3] / (self.purchases_item1[2][3] + self.purchases_item21[2][3])

        # purchase2 and purchase1 per class / purchase1 per class
        if self.purchases_item1[3][0] + self.purchases_item21[3][0]:
            self.conversion_rates_item21[3][0] = self.purchases_item21[3][0] / (self.purchases_item1[3][0] + self.purchases_item21[3][0])
        if self.purchases_item1[3][1] + self.purchases_item21[3][1]:
            self.conversion_rates_item21[3][1] = self.purchases_item21[3][1] / (self.purchases_item1[3][1] + self.purchases_item21[3][1])
        if self.purchases_item1[3][2] + self.purchases_item21[3][2]:
            self.conversion_rates_item21[3][2] = self.purchases_item21[3][2] / (self.purchases_item1[3][2] + self.purchases_item21[3][2])
        if self.purchases_item1[3][3] + self.purchases_item21[3][3]:
            self.conversion_rates_item21[3][3] = self.purchases_item21[3][3] / (self.purchases_item1[3][3] + self.purchases_item21[3][3])


#Pc1(I2| I1 and P0) = P(I2, I1, P0) / P(I1 and P0)