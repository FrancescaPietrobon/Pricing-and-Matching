import numpy as np
import math
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

    def round(self, pulled_arm):
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

        conversion_rates_item1_round = np.zeros(4)
        for i in range(4):
            conversion_rates_item1_round[i] = buyers_item1[i] / offer_item1[i] if offer_item1[i] > 0 else 0

        conversion_rates_item2_round = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                conversion_rates_item2_round[i, j] = buyers_item2[i, j] / offer_item2[i, j] if offer_item2[i, j] > 0 else 0

        selected_margin_item1 = self.margins_item1[pulled_arm]
        revenue_item1 = 0
        revenue_item1 = revenue_item1 + buyers_item1.sum() * selected_margin_item1

        revenue_item2 = 0
        for promo_type in range(4):
            revenue_item2 = revenue_item2 + (buyers_item2[promo_type]).sum() * self.margin_item2 * (1-self.discounts[promo_type])

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

    def round(self, weights):
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

        conversion_rates_item1_round = np.zeros(4)
        for i in range(4):
            conversion_rates_item1_round[i] = buyers_item1[i] / offer_item1[i] if offer_item1[i] > 0 else 0

        conversion_rates_item2_round = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                conversion_rates_item2_round[i, j] = buyers_item2[i, j] / offer_item2[i, j] if offer_item2[i, j] > 0 else 0

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

    def round(self, pulled_arm):
        # Normalizing the weights matrix to have proper values between 0 and 1
        weights = np.zeros((4, 4))
        for class_type in range(4):
            weights[:, class_type] = pulled_arm[1][:, class_type] / pulled_arm[1][:, class_type].sum()

        # Simulating the arrival of customers
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

        conversion_rates_item1_round = np.zeros(4)
        for i in range(4):
            conversion_rates_item1_round[i] = buyers_item1[i] / offer_item1[i] if offer_item1[i] > 0 else 0

        conversion_rates_item2_round = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                conversion_rates_item2_round[i, j] = buyers_item2[i, j] / offer_item2[i, j] if offer_item2[i, j] > 0 else 0

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

    def sample(self):
        return np.clip(np.random.normal(self.mean, self.sd), 0, 500).astype(int)

########################################################################################################################


class Non_Stationary_Environment_First:
    def __init__(self, n_arms, probabilities, horizon):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.t = 0
        n_phases = np.shape(self.probabilities)[0]
        self.phases_size = horizon/n_phases

    def round(self, pulled_arm):
        current_phase = int(self.t / self.phases_size)
        p = self.probabilities[current_phase, pulled_arm]
        reward = np.random.binomial(1, p)
        self.t += 1
        return reward


class Non_Stationary_Environment_Third:
    def __init__(self, n_arms, probabilities, horizon):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.time = 0
        n_phases = np.shape(self.probabilities)[0]
        self.phases_size = int(horizon/n_phases)

    def round(self, pulled_arm):
        current_phase = math.floor(self.time / self.phases_size)
        p = self.probabilities[current_phase, :, pulled_arm]
        reward = np.random.binomial(1, p)
        self.time += 1
        return reward
