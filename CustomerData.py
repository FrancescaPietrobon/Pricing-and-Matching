class CustomerData:
    def __init__(self, identifier, group):
        self.identifier = identifier                # Unique identifier for the customer
        self.group = group                          # Group type of the customer
        self.item1_purchased = False                # True if the customer bought the first item
        self.item2_purchased = False                # True if the customer bought the second item
        self.p0_received = False                    # True if promo P0 is received
        self.p1_received = False                    # True if promo P1 is received
        self.p2_received = False                    # True if promo P2 is received
        self.p3_received = False                    # True if promo P3 is received

# Getter methods
    def get_identifier(self):
        return self.identifier

    def get_group(self):
        return self.group

    def purchased_item1(self):
        return self.item1_purchased

    def purchased_item2(self):
        return self.item2_purchased

    def received_p0(self):
        return self.p0_received

    def received_p1(self):
        return self.p1_received

    def received_p2(self):
        return self.p2_received

    def received_p3(self):
        return self.p3_received

# Setter methods
    def buy_item1(self):
        self.item1_purchased = True

    def buy_item2(self):
        self.item2_purchased = True

    def give_p0(self):
        self.p0_received = True

    def give_p1(self):
        self.p1_received = True

    def give_p2(self):
        self.p2_received = True

    def give_p3(self):
        self.p3_received = True
