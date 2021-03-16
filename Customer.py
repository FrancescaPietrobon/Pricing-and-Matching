class Customer:
    def __init__(self, identifier, group):
        self.identifier = identifier
        self.first_purchase = None
        self.second_purchase = None
        self.first_promo = False
        self.second_promo = False
        self.third_promo = False
        self.fourth_promo = False

        self.group = group

    def get_first_purchase(self):
        return self.first_purchase

    def get_second_purchase(self):
        return self.second_purchase

    def is_first_promo(self):
        return self.first_promo

    def is_second_promo(self):
        return self.second_promo

    def is_third_promo(self):
        return self.third_promo

    def is_fourth_promo(self):
        return self.fourth_promo

    def set_first_purchase(self, item):
        self.first_purchase = item

    def set_second_purchase(self, item):
        self.second_purchase = item

    def set_true_first_promo(self):
        self.first_promo = True

    def set_true_second_promo(self):
        self.second_promo = True

    def set_true_third_promo(self):
        self.third_promo = True

    def set_true_fourth_promo(self):
        self.fourth_promo = True
