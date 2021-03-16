class CustomerData:
    def __init__(self, customer):
        self.customer = customer
        self.first_purchase = False
        self.second_purchase = False
        self.first_promo = False            # first_promo = True if it has been given in the past
        self.second_promo = False           # same
        self.third_promo = False            # same
        self.fourth_promo = False           # same

    def is_first_purchase(self):
        return self.first_purchase

    def is_second_purchase(self):
        return self.second_purchase

    def is_first_promo(self):
        return self.first_promo

    def is_second_promo(self):
        return self.second_promo

    def is_third_promo(self):
        return self.third_promo

    def is_fourth_promo(self):
        return self.fourth_promo

    def set_true_first_purchase(self):
        self.first_purchase = True

    def set_true_second_purchase(self):
        self.second_purchase = True

    def set_true_first_promo(self):
        self.first_promo = True

    def set_true_second_promo(self):
        self.second_promo = True

    def set_true_third_promo(self):
        self.third_promo = True

    def set_true_fourth_promo(self):
        self.fourth_promo = True
