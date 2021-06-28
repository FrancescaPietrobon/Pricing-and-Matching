import numpy as np
np.random.seed(1234)


class Environment_Single_Price:
    def __init__(self, margins_item1, margin_item2, conversion_rates_item1, conversion_rates_item2, weights, daily_customers, discounts):
        self.margins_item1 = margins_item1
        self.margin_item2 = margin_item2
        self.conversion_rates_item1 = conversion_rates_item1
        self.conversion_rates_item2 = conversion_rates_item2
        self.weights = weights
        self.daily_customers = daily_customers
        self.discounts = discounts

    # The pulled_arm input represents the margin selected by the Bandit
    def round(self, pulled_arm):
        # Simulating the arrival of customers, one by one. A customer of a specific class arrives, and with a certain
        # probability he/she will buy the first item (binomial using its conversion rate as parameter).
        # If he/she bought it, then he/she receives a promo according to the matching weights matrix.
        # Finally, he/she can buy the second item (binomial using its conversion rate as parameter).
        buyers_item1 = np.zeros(4)
        buyers_item2 = np.zeros((4, 4))
        offer_item1 = np.zeros(4)
        offer_item2 = np.zeros((4, 4))
        for group in range(4):
            for _ in range(self.daily_customers[group]):
                bin_item1 = np.random.binomial(1, self.conversion_rates_item1[group, pulled_arm])
                offer_item1[group] += 1
                buyers_item1[group] = buyers_item1[group] + bin_item1
                if bin_item1 == 1:
                    promo = np.random.choice(4, p=self.weights[:, group])
                    offer_item2[promo, group] += 1
                    buyers_item2[promo, group] = buyers_item2[promo, group] + np.random.binomial(1, self.conversion_rates_item2[promo, group])

        # Given the observed arrivals and behaviors above, we can compute the conversion rates for the first item,
        # as number of customers who bought the first item over number of customers who could have bought it (offer)
        conversion_rates_item1_round = np.zeros(4)
        for class_type in range(4):
            conversion_rates_item1_round[class_type] = buyers_item1[class_type] / offer_item1[class_type] if offer_item1[class_type] > 0 else 0

        # We can do the same for the conversion rates for the second item
        conversion_rates_item2_round = np.zeros((4, 4))
        for promo_type in range(4):
            for class_type in range(4):
                conversion_rates_item2_round[promo_type, class_type] = buyers_item2[promo_type, class_type] / offer_item2[promo_type, class_type] if offer_item2[promo_type, class_type] > 0 else 0

        # Finally, we can compute the total revenue for the day, using the known price for the second item
        # and the pulled price for the first item
        selected_margin_item1 = self.margins_item1[pulled_arm]
        revenue_item1 = 0
        revenue_item1 += buyers_item1.sum() * selected_margin_item1

        revenue_item2 = 0
        for promo_type in range(4):
            revenue_item2 += (buyers_item2[promo_type]).sum() * self.margin_item2 * (1-self.discounts[promo_type])

        revenue = revenue_item1 + revenue_item2

        return conversion_rates_item1_round, conversion_rates_item2_round, revenue


class Environment_Matching:
    def __init__(self, margin_item1, margin_item2, conversion_rates_item1, conversion_rates_item2, daily_customers, discounts):
        self.margin_item1 = margin_item1
        self.margin_item2 = margin_item2
        self.conversion_rates_item1 = conversion_rates_item1
        self.conversion_rates_item2 = conversion_rates_item2
        self.daily_customers = daily_customers
        self.discounts = discounts

    # The weights input is the normalized matching weights matrix computed by the LP
    def round(self, weights):
        # Simulating the arrival of customers, one by one. A customer of a specific class arrives, and with a certain
        # probability he/she will buy the first item (binomial using its conversion rate as parameter).
        # If he/she bought it, then he/she receives a promo according to the matching weights matrix.
        # Finally, he/she can buy the second item (binomial using its conversion rate as parameter).
        buyers_item1 = np.zeros(4)
        buyers_item2 = np.zeros((4, 4))
        offer_item1 = np.zeros(4)
        offer_item2 = np.zeros((4, 4))
        for group in range(4):
            for _ in range(self.daily_customers[group]):
                bin_item1 = np.random.binomial(1, self.conversion_rates_item1[group])
                offer_item1[group] += 1
                buyers_item1[group] = buyers_item1[group] + bin_item1
                if bin_item1 == 1:
                    promo = np.random.choice(4, p=weights[:, group])
                    offer_item2[promo, group] += 1
                    buyers_item2[promo, group] = buyers_item2[promo, group] + np.random.binomial(1, self.conversion_rates_item2[promo, group])

        # Given the observed arrivals and behaviors above, we can compute the conversion rates for the first item,
        # as number of customers who bought the first item over number of customers who could have bought it (offer)
        conversion_rates_item1_round = np.zeros(4)
        for class_type in range(4):
            conversion_rates_item1_round[class_type] = buyers_item1[class_type] / offer_item1[class_type] if offer_item1[class_type] > 0 else 0

        # We can do the same for the conversion rates for the second item
        conversion_rates_item2_round = np.zeros((4, 4))
        for promo_type in range(4):
            for class_type in range(4):
                conversion_rates_item2_round[promo_type, class_type] = buyers_item2[promo_type, class_type] / offer_item2[promo_type, class_type] if offer_item2[promo_type, class_type] > 0 else 0

        # Finally, we can compute the total revenue for the day, using the known prices for the two items
        revenue_item1 = 0
        revenue_item1 = revenue_item1 + buyers_item1.sum() * self.margin_item1

        revenue_item2 = 0
        for promo_type in range(4):
            revenue_item2 = revenue_item2 + (buyers_item2[promo_type]).sum() * self.margin_item2 * (1-self.discounts[promo_type])

        revenue = revenue_item1 + revenue_item2

        return conversion_rates_item1_round, conversion_rates_item2_round, revenue


class Environment_Double_Prices_Matching:
    def __init__(self, margins_item1, margins_item2, conversion_rates_item1, conversion_rates_item2, daily_customers, discounts, promo_fractions):
        self.margins_item1 = margins_item1
        self.margins_item2 = margins_item2
        self.conversion_rates_item1 = conversion_rates_item1
        self.conversion_rates_item2 = conversion_rates_item2
        self.daily_customers = daily_customers
        self.discounts = discounts
        self.promo_fractions = promo_fractions

    # The pulled_arm input is composed of the two margins selected by the Bandit (pulled_arm[0][0] and pulled_arm[0][1]
    # respectively for the two items) and the matching matrix, not normalized
    def round(self, pulled_arm):
        # Normalizing the weights matrix to have proper values between 0 and 1
        weights = np.zeros((4, 4))
        for class_type in range(4):
            weights[:, class_type] = pulled_arm[1][:, class_type] / pulled_arm[1][:, class_type].sum() if np.any(weights[:, class_type]) else np.full(4, 0.25)

        # Simulating the arrival of customers, one by one. A customer of a specific class arrives, and with a certain
        # probability he/she will buy the first item (binomial using its conversion rate as parameter).
        # If he/she bought it, then he/she receives a promo according to the matching weights matrix.
        # Finally, he/she can buy the second item (binomial using its conversion rate as parameter).
        buyers_item1 = np.zeros(4)
        buyers_item2 = np.zeros((4, 4))
        offer_item1 = np.zeros(4)
        offer_item2 = np.zeros((4, 4))
        for group in range(4):
            for _ in range(self.daily_customers[group]):
                bin_item1 = np.random.binomial(1, self.conversion_rates_item1[group, pulled_arm[0][0]])
                offer_item1[group] += 1
                buyers_item1[group] = buyers_item1[group] + bin_item1
                if bin_item1 == 1:
                    promo = np.random.choice(4, p=weights[:, group])
                    offer_item2[promo, group] += 1
                    buyers_item2[promo, group] = buyers_item2[promo, group] + np.random.binomial(1, self.conversion_rates_item2[pulled_arm[0][1], promo, group])

        # Given the observed arrivals and behaviors above, we can compute the conversion rates for the first item,
        # as number of customers who bought the first item over number of customers who could have bought it (offer)
        conversion_rates_item1_round = np.zeros(4)
        for class_type in range(4):
            conversion_rates_item1_round[class_type] = buyers_item1[class_type] / offer_item1[class_type] if offer_item1[class_type] > 0 else 0

        # We can do the same for the conversion rates for the second item
        conversion_rates_item2_round = np.zeros((4, 4))
        for promo_type in range(4):
            for class_type in range(4):
                conversion_rates_item2_round[promo_type, class_type] = buyers_item2[promo_type, class_type] / offer_item2[promo_type, class_type] if offer_item2[promo_type, class_type] > 0 else 0

        # Finally, we can compute the total revenue for the day, using the pulled prices for the two items
        selected_margin_item1 = self.margins_item1[pulled_arm[0][0]]
        revenue_item1 = 0
        revenue_item1 = revenue_item1 + buyers_item1.sum() * selected_margin_item1

        selected_margin_item2 = self.margins_item2[pulled_arm[0][1]]
        revenue_item2 = 0
        for promo_type in range(4):
            revenue_item2 = revenue_item2 + (buyers_item2[promo_type]).sum() * selected_margin_item2 * (1-self.discounts[promo_type])

        revenue = revenue_item1 + revenue_item2

        return conversion_rates_item1_round, conversion_rates_item2_round, revenue


class Non_Stationary_Environment(Environment_Double_Prices_Matching):
    def __init__(self, margins_item1, margins_item2, conversion_rates_item1_NS, conversion_rates_item2_NS, daily_customers, discounts, promo_fractions, phases_len):
        super().__init__(margins_item1, margins_item2, conversion_rates_item1_NS, conversion_rates_item2_NS, daily_customers, discounts, promo_fractions)
        self.t = 0
        self.conversion_rates_item1_NS = conversion_rates_item1_NS
        self.conversion_rates_item2_NS = conversion_rates_item2_NS
        self.phases_len = phases_len

    # Here, the Environment_Double_Prices_Matching class is used, just selecting the right conversion rates for the two
    # items according to the current phase
    def round(self, pulled_arm):
        current_phase = int(self.t / self.phases_len)
        self.conversion_rates_item1 = self.conversion_rates_item1_NS[current_phase]
        self.conversion_rates_item2 = self.conversion_rates_item2_NS[current_phase]
        self.t += 1
        return super().round(pulled_arm)


class Daily_Customers:
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    # The daily customer sample for the day is extracted using a Gaussian distribution
    def sample(self):
        return np.clip(np.random.normal(self.mean, self.sd), 0, 500).astype(int)
