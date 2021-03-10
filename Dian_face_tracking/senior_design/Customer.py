import numpy as np

#TODO update likelihoods
#TODO add more menu items
#TODO potentially change customer structure

class Customer_Order():
    def __init__(self, item_name = None, item_amount = None, order = None):
        self.item_name = item_name
        self.item_amount = item_amount
        self.order_list = ["Chicken Sandwich", "Deluxe Sandwich", "Spicy Chicken Sandwich", "Spicy Deluxe Sandwich",
            "Grilled Chicken Sandwich", "Grilled Chicken Club", "Nuggets", "Chick-n-Strips",
            "Grilled Cool Wrap", "Grilled Nuggets"]
        self.order = order
        # if no dict is passed in
        if self.order is None:
            if (item not in self.order_list for item in item_name) :
                raise ValueError("bad order", item)
            self.order = {(key, value) for (key, value) in zip(self.order_list, self.item_amount)}                   
    
    def __str__(self):
        return ("".join(str(key) + ": " + str(value) for key, value in self.order.items()))
    


class Customer():
    def __init__(self, past_order = None, current_order = None, likelihoods = None, faceid = None):
        # {"Chicken Sandwich" : (how many)}
        if past_order is None:
            self.past_orders = Customer_Order(order = current_order)

        else:
            self.past_orders = past_order
        # [3%, 10%]
        self.current_orders = Customer_Order(order = current_order)
        self.total_orders = 0
        self.item_ordering_likelihood = likelihoods
        self.faceID = faceid
    
    def add(self, order):
        # add order
        for item in order:
            self.past_orders.add(item)
        # update likelihood 
    
    def __str__(self):
        return("faceID: {}\ncurrent_orders: {}".format(self.faceID, self.current_orders))

if __name__ == "__main__":
    order = Customer_Order(["Chick_Sandwich"],[2])
    print("{}: {}".format(order.item_name, order.item_amount))
    order2 = Customer_Order(order = {"Delux Sandwich": 3})
    print(order2)
    customer1 = Customer({"Deluxe Sandwich": 3})
    print(customer1)