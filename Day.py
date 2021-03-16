class Day:
    def __init__(self, identifier):
        self.identifier = identifier            # Identifier of the day
        self.number_of_customers = 0            # Number of customers per day
        self.customers_data_list = []           # List of CustomerData objects

    def get_id(self):
        return self.identifier

    def add_customer_data(self, customer_data):
        self.customers_data_list.append(customer_data)          # Adding the CustomerData object to the list
        self.number_of_customers += 1                           # Incrementing the number of customers for the day

    def get_number_of_customers(self):
        return self.number_of_customers

    def get_customers_data_list(self):
        return self.customers_data_list
