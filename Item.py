class Item:
    def __init__(self, name, price, margin):
        self.name = name
        self.price = price
        self.margin = margin

    def get_name(self):
        return self.name

    def get_price(self):
        return self.price

    def get_margin(self):
        return self.margin
