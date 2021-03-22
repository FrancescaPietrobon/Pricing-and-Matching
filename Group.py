class Group:
    def __init__(self, number, conversion_rate_item1=0.5, conversion_rate_item2=0.5, conversion_rate_item2_given_item1=0.5):
        self.number = number
        self.conversion_rate_item1 = conversion_rate_item1
        self.conversion_rate_item2 = conversion_rate_item2
        self.conversion_rate_item2_given_item1 = conversion_rate_item2_given_item1

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

    def get_conversion_rate_item2_given_item1(self):
        return self.conversion_rate_item2_given_item1

    def set_conversion_rate_item2_given_item1(self, value):
        self.conversion_rate_item2_given_item1 = value
