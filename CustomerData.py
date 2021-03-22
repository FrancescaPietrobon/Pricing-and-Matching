class CustomerData:
    def __init__(self, identifier, group):
        self.identifier = identifier            # Unique identifier for the customer
        self.group = group                      # Group type of the customer
        self.first_item_purchase = False
        self.second_item_purchase = False
        self.first_type_promo = False           # True if P0 is received
        self.second_type_promo = False          # same
        self.third_type_promo = False           # same
        self.fourth_type_promo = False          # same

    def get_identifier(self):
        return self.identifier

    def get_group(self):
        return self.group

    def is_first_purchase(self):
        return self.first_item_purchase

    def is_second_purchase(self):
        return self.second_item_purchase

    def is_first_promo(self):
        return self.first_type_promo

    def is_second_promo(self):
        return self.second_type_promo

    def is_third_promo(self):
        return self.third_type_promo

    def is_fourth_promo(self):
        return self.fourth_type_promo

    def set_true_first_purchase(self):
        self.first_item_purchase = True

    def set_true_second_purchase(self):
        self.second_item_purchase = True

    def set_true_first_promo(self):
        self.first_type_promo = True

    def set_true_second_promo(self):
        self.second_type_promo = True

    def set_true_third_promo(self):
        self.third_type_promo = True

    def set_true_fourth_promo(self):
        self.fourth_type_promo = True
