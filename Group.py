class Group:
    def __init__(self, number, conversion_rate_item1=0, conversion_rate_item2=0,
                 conversion_rate_item2_given_item1_P0=0,
                 conversion_rate_item2_given_item1_P1=0,
                 conversion_rate_item2_given_item1_P2=0,
                 conversion_rate_item2_given_item1_P3=0):
        self.number = number
        self.conversion_rate_item1 = conversion_rate_item1
        self.conversion_rate_item2 = conversion_rate_item2
        self.conversion_rate_item2_given_item1_P0 = conversion_rate_item2_given_item1_P0
        self.conversion_rate_item2_given_item1_P1 = conversion_rate_item2_given_item1_P1
        self.conversion_rate_item2_given_item1_P2 = conversion_rate_item2_given_item1_P2
        self.conversion_rate_item2_given_item1_P3 = conversion_rate_item2_given_item1_P3

    def get_number(self):
        return self.number

    def get_conversion_rate_item1(self):
        return self.conversion_rate_item1

    def set_conversion_rate_item1(self, value):
        self.conversion_rate_item1 = value

    def get_conversion_rate_item2(self):
        return self.conversion_rate_item2

    def set_conversion_rate_item2(self, value):
        self.conversion_rate_item2 = value

    def get_conversion_rate_item2_given_item1_P0(self):
        return self.conversion_rate_item2_given_item1_P0

    def set_conversion_rate_item2_given_item1_P0(self, value):
        self.conversion_rate_item2_given_item1_P0 = value

    def get_conversion_rate_item2_given_item1_P1(self):
        return self.conversion_rate_item2_given_item1_P1

    def set_conversion_rate_item2_given_item1_P1(self, value):
        self.conversion_rate_item2_given_item1_P1 = value

    def get_conversion_rate_item2_given_item1_P2(self):
        return self.conversion_rate_item2_given_item1_P2

    def set_conversion_rate_item2_given_item1_P2(self, value):
        self.conversion_rate_item2_given_item1_P2 = value

    def get_conversion_rate_item2_given_item1_P3(self):
        return self.conversion_rate_item2_given_item1_P3

    def set_conversion_rate_item2_given_item1_P3(self, value):
        self.conversion_rate_item2_given_item1_P3 = value
