import matching_lp as lp
from sklearn.preprocessing import normalize
import numpy as np
np.random.seed(1234)

class Matching:
    def __init__(self, margin_item2, conversion_rates_item1, conversion_rates_item2, daily_customers, discounts, promo_fractions):
        self.margin_item2 = margin_item2
        self.conversion_rates_item1 = conversion_rates_item1
        self.conversion_rates_item2 = conversion_rates_item2
        self.daily_customers = daily_customers
        self.discounts = discounts
        self.promo_fractions = promo_fractions

    def optimize(self, t):
        if t == 0:
            matching = np.random.rand(4, 4)
        else:
            daily_promos = (self.promo_fractions * sum(self.daily_customers * self.conversion_rates_item1)).astype(int)
            _, matching = lp.matching_lp(self.margin_item2, self.discounts, self.conversion_rates_item2,
                                         daily_promos, (self.daily_customers * self.conversion_rates_item1).astype(int))

        return normalize(matching, 'l1', axis=0)
