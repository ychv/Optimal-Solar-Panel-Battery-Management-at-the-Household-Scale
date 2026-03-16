"""
Utility functions for environnement definition
"""
import numpy as np

def update_conso(conso):
    conso = np.array(conso)
    return conso + np.random.randint(0,2,conso.size)

def update_prod(prod):
    prod = np.array(prod)
    return prod + np.random.randint(0,2,prod.size)

def update_price(price):
    price = np.array(price)
    return price + np.random.randint(0,2,price.size)