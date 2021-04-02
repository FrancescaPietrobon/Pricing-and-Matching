class Day:                                      #NB we do not care about peole who do not buy anything becasue our aim is to maximize purchases of item2 given item1
    def __init__(self, identifier, group1, group2, group3, group4):
        self.identifier = identifier            # Identifier of the day
        self.customers_data_list = []           # List of CustomerData objects
        self.group1 = group1
        self.group2 = group2
        self.group3 = group3
        self.group4 = group4

        #statistics of the day

        self.number_of_customers = 0            # Number of customers per day
        self.number_of_c1 = 0                   # Number of customers per day of class 1
        self.number_of_c2 = 0
        self.number_of_c3 = 0
        self.number_of_c4 = 0

        self.number_of_item2_c1 = 0                # Number of customers of class 1 purchasing 2 in general
        self.number_of_item2_c2 = 0
        self.number_of_item2_c3 = 0
        self.number_of_item2_c4 = 0

        self.purchase_item1_P0_c1 = 0              # Number of customer of class 1 purchasing only 1 with P0
        self.purchase_item1_P0_c2 = 0
        self.purchase_item1_P0_c3 = 0
        self.purchase_item1_P0_c4 = 0

        self.purchase_item1_P1_c1 = 0              # Number of customer of class 1 purchasing only 1 with P1
        self.purchase_item1_P1_c2 = 0
        self.purchase_item1_P1_c3 = 0
        self.purchase_item1_P1_c4 = 0

        self.purchase_item1_P2_c1 = 0              # Number of customer of class 1 purchasing only 1 with P2
        self.purchase_item1_P2_c2 = 0
        self.purchase_item1_P2_c3 = 0
        self.purchase_item1_P2_c4 = 0

        self.purchase_item1_P3_c1 = 0              # Number of customer of class 1 purchasing only 1 with P3
        self.purchase_item1_P3_c2 = 0
        self.purchase_item1_P3_c3 = 0
        self.purchase_item1_P3_c4 = 0

        self.purchase_item2_given1_P0_c1 = 0       # Number of customer of class one buying item 2 given P0
        self.purchase_item2_given1_P0_c2 = 0
        self.purchase_item2_given1_P0_c3 = 0
        self.purchase_item2_given1_P0_c4 = 0

        self.purchase_item2_given1_P1_c1 = 0       # Number of customer of class one buying item 2 given P1
        self.purchase_item2_given1_P1_c2 = 0
        self.purchase_item2_given1_P1_c3 = 0
        self.purchase_item2_given1_P1_c4 = 0

        self.purchase_item2_given1_P2_c1 = 0       # Number of customer of class one buying item 2 given P2
        self.purchase_item2_given1_P2_c2 = 0
        self.purchase_item2_given1_P2_c3 = 0
        self.purchase_item2_given1_P2_c4 = 0

        self.purchase_item2_given1_P3_c1 = 0       # Number of customer of class one buying item 2 given P3
        self.purchase_item2_given1_P3_c2 = 0
        self.purchase_item2_given1_P3_c3 = 0
        self.purchase_item2_given1_P3_c4 = 0

    def get_id(self):
        return self.identifier

    def add_customer_data(self, customer_data):
        self.customers_data_list.append(customer_data)          # Adding the CustomerData object to the list
        self.number_of_customers += 1                           # Incrementing the number of customers for the day
        if customer_data.is_second_purchase():
            if customer_data.get_group() == 1:
                self.number_of_item2_c1 += 1
            elif customer_data.get_group() == 2:
                self.number_of_item2_c2 += 1
            elif customer_data.get_group() == 3:
                self.number_of_item2_c3 += 1
            elif customer_data.get_group() == 4:
                self.number_of_item2_c4 += 1
        if customer_data.is_first_purchase():
            if customer_data.get_group() == 1:
                self.number_of_c1 += 1
                if customer_data.is_first_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P0_c1 += 1
                    else:
                        self.purchase_item1_P0_c1 += 1
                elif customer_data.is_second_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P1_c1 += 1
                    else:
                        self.purchase_item1_P1_c1 += 1
                elif customer_data.is_third_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P2_c1 += 1
                    else:
                        self.purchase_item1_P2_c1 += 1
                elif customer_data.is_fourth_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P3_c1 += 1
                    else:
                        self.purchase_item1_P3_c1 += 1


            elif customer_data.get_group() == 2:
                self.number_of_c2 += 1
                if customer_data.is_first_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P0_c2 += 1
                    else:
                        self.purchase_item1_P0_c2 += 1
                elif customer_data.is_second_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P1_c2 += 1
                    else:
                        self.purchase_item1_P1_c2 += 1
                elif customer_data.is_third_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P2_c2 += 1
                    else:
                        self.purchase_item1_P2_c2 += 1
                elif customer_data.is_fourth_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P3_c2 += 1
                    else:
                        self.purchase_item1_P3_c2 += 1

            elif customer_data.get_group() == 3:
                self.number_of_c3 += 1
                if customer_data.is_first_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P0_c3 += 1
                    else:
                        self.purchase_item1_P0_c3 += 1
                elif customer_data.is_second_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P1_c3 += 1
                    else:
                        self.purchase_item1_P1_c3 += 1
                elif customer_data.is_third_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P2_c3 += 1
                    else:
                        self.purchase_item1_P2_c3 += 1
                elif customer_data.is_fourth_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P3_c3 += 1
                    else:
                        self.purchase_item1_P3_c3 += 1

            elif customer_data.get_group() == 4:
                self.number_of_c4 += 1
                if customer_data.is_first_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P0_c4 += 1
                    else:
                        self.purchase_item1_P0_c4 += 1
                elif customer_data.is_second_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P1_c4 += 1
                    else:
                        self.purchase_item1_P1_c4 += 1
                elif customer_data.is_third_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P2_c4 += 1
                    else:
                        self.purchase_item1_P2_c4 += 1
                elif customer_data.is_fourth_promo():
                    if customer_data.is_second_purchase():
                        self.purchase_item2_given1_P3_c4 += 1
                    else:
                        self.purchase_item1_P3_c4 += 1


    def get_number_of_customers(self):
        return self.number_of_customers

    def get_number_of_customers_c1(self):
        return self.number_of_c1

    def get_number_of_customers_c2(self):
        return self.number_of_c2

    def get_number_of_customers_c3(self):
        return self.number_of_c3

    def get_number_of_customers_c4(self):
        return self.number_of_c4

    def get_customers_data_list(self):
        return self.customers_data_list

    def set_conversion_rate(self):
        # purchase1 per class / total_purchase per class
        if self.number_of_c1 > 0:
            self.group1.set_conversion_rate_item1((self.purchase_item1_P0_c1 + self.purchase_item1_P1_c1 + self.purchase_item1_P2_c1 + self.purchase_item1_P3_c1 + self.purchase_item2_given1_P0_c1 + self.purchase_item2_given1_P1_c1 + self.purchase_item2_given1_P2_c1 + self.purchase_item2_given1_P3_c1)/self.number_of_c1)
        if self.number_of_c2 > 0:
            self.group2.set_conversion_rate_item1((self.purchase_item1_P0_c2 + self.purchase_item1_P1_c2 + self.purchase_item1_P2_c2 + self.purchase_item1_P3_c2 + self.purchase_item2_given1_P0_c2 + self.purchase_item2_given1_P1_c2 + self.purchase_item2_given1_P2_c2 + self.purchase_item2_given1_P3_c2)/self.number_of_c2)
        if self.number_of_c3 > 0:
            self.group3.set_conversion_rate_item1((self.purchase_item1_P0_c3 + self.purchase_item1_P1_c3 + self.purchase_item1_P2_c3 + self.purchase_item1_P3_c3 + self.purchase_item2_given1_P0_c3 + self.purchase_item2_given1_P1_c3 + self.purchase_item2_given1_P2_c3 + self.purchase_item2_given1_P3_c3)/self.number_of_c3)
        if self.number_of_c4 > 0:
            self.group4.set_conversion_rate_item1((self.purchase_item1_P0_c4 + self.purchase_item1_P1_c4 + self.purchase_item1_P2_c4 + self.purchase_item1_P3_c4 + self.purchase_item2_given1_P0_c4 + self.purchase_item2_given1_P1_c4 + self.purchase_item2_given1_P2_c4 + self.purchase_item2_given1_P3_c4)/self.number_of_c4)

        # purchase2 per class / total_purchase per class
        if self.number_of_c1 > 0:
            self.group1.set_conversion_rate_item2(self.number_of_item2_c1/self.number_of_c1)
        if self.number_of_c2 > 0:
            self.group2.set_conversion_rate_item2(self.number_of_item2_c2/self.number_of_c2)
        if self.number_of_c3 > 0:
            self.group2.set_conversion_rate_item2(self.number_of_item2_c3/self.number_of_c3)
        if self.number_of_c4 > 0:
            self.group2.set_conversion_rate_item2(self.number_of_item2_c4/self.number_of_c3)

        # purchase2 and purchase1 per class / purchase1 per class
        if (self.purchase_item1_P0_c1 + self.purchase_item2_given1_P0_c1) > 0:
            self.group1.set_conversion_rate_item2_given_item1_P0(self.purchase_item2_given1_P0_c1 / (self.purchase_item1_P0_c1 + self.purchase_item2_given1_P0_c1))
        if (self.purchase_item1_P0_c2 + self.purchase_item2_given1_P0_c2) > 0:
            self.group2.set_conversion_rate_item2_given_item1_P0(self.purchase_item2_given1_P0_c2 / (self.purchase_item1_P0_c2 + self.purchase_item2_given1_P0_c2))
        if (self.purchase_item1_P0_c3 + self.purchase_item2_given1_P0_c3) > 0:
            self.group3.set_conversion_rate_item2_given_item1_P0(self.purchase_item2_given1_P0_c3 / (self.purchase_item1_P0_c3 + self.purchase_item2_given1_P0_c3))
        if (self.purchase_item1_P0_c4 + self.purchase_item2_given1_P0_c4) > 0:
            self.group4.set_conversion_rate_item2_given_item1_P0(self.purchase_item2_given1_P0_c4 / (self.purchase_item1_P0_c4 + self.purchase_item2_given1_P0_c4))

        # purchase2 and purchase1 per class / purchase1 per class
        if (self.purchase_item1_P1_c1 + self.purchase_item2_given1_P1_c1) > 0:
            self.group1.set_conversion_rate_item2_given_item1_P1(self.purchase_item2_given1_P1_c1 / (self.purchase_item1_P1_c1 + self.purchase_item2_given1_P1_c1))
        if (self.purchase_item1_P1_c2 + self.purchase_item2_given1_P1_c2) > 0:
            self.group2.set_conversion_rate_item2_given_item1_P1(self.purchase_item2_given1_P1_c2 / (self.purchase_item1_P1_c2 + self.purchase_item2_given1_P1_c2))
        if (self.purchase_item1_P1_c3 + self.purchase_item2_given1_P1_c3) > 0:
            self.group3.set_conversion_rate_item2_given_item1_P1(self.purchase_item2_given1_P1_c3 / (self.purchase_item1_P1_c3 + self.purchase_item2_given1_P1_c3))
        if (self.purchase_item1_P1_c4 + self.purchase_item2_given1_P1_c4) > 0:
            self.group4.set_conversion_rate_item2_given_item1_P1(self.purchase_item2_given1_P1_c4 / (self.purchase_item1_P1_c4 + self.purchase_item2_given1_P1_c4))

        # purchase2 and purchase1 per class / purchase1 per class
        if (self.purchase_item1_P2_c1 + self.purchase_item2_given1_P2_c1) > 0:
            self.group1.set_conversion_rate_item2_given_item1_P2(self.purchase_item2_given1_P2_c1 / (self.purchase_item1_P2_c1 + self.purchase_item2_given1_P2_c1))
        if (self.purchase_item1_P2_c2 + self.purchase_item2_given1_P2_c2) > 0:
            self.group2.set_conversion_rate_item2_given_item1_P2(self.purchase_item2_given1_P2_c2 / (self.purchase_item1_P2_c2 + self.purchase_item2_given1_P2_c2))
        if (self.purchase_item1_P2_c3 + self.purchase_item2_given1_P2_c3) > 0:
            self.group3.set_conversion_rate_item2_given_item1_P2(self.purchase_item2_given1_P2_c3 / (self.purchase_item1_P2_c3 + self.purchase_item2_given1_P2_c3))
        if (self.purchase_item1_P2_c4 + self.purchase_item2_given1_P2_c4) > 0:
            self.group4.set_conversion_rate_item2_given_item1_P2(self.purchase_item2_given1_P2_c4 / (self.purchase_item1_P2_c4 + self.purchase_item2_given1_P2_c4))

        # purchase2 and purchase1 per class / purchase1 per class
        if (self.purchase_item1_P3_c1 + self.purchase_item2_given1_P3_c1):
            self.group1.set_conversion_rate_item2_given_item1_P3(self.purchase_item2_given1_P3_c1 / (self.purchase_item1_P3_c1 + self.purchase_item2_given1_P3_c1))
        if (self.purchase_item1_P3_c2 + self.purchase_item2_given1_P3_c2):
            self.group2.set_conversion_rate_item2_given_item1_P3(self.purchase_item2_given1_P3_c2 / (self.purchase_item1_P3_c2 + self.purchase_item2_given1_P3_c2))
        if (self.purchase_item1_P3_c3 + self.purchase_item2_given1_P3_c3):
            self.group3.set_conversion_rate_item2_given_item1_P3(self.purchase_item2_given1_P3_c3 / (self.purchase_item1_P3_c3 + self.purchase_item2_given1_P3_c3))
        if (self.purchase_item1_P3_c4 + self.purchase_item2_given1_P3_c4):
            self.group4.set_conversion_rate_item2_given_item1_P3(self.purchase_item2_given1_P3_c4 / (self.purchase_item1_P3_c4 + self.purchase_item2_given1_P3_c4))


#Pc1(I2| I1 and P0) = P(I2, I1, P0) / P(I1 and P0)