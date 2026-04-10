"""
Class definition for price model
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class PrixDay:
    def __init__(self,pas_t=15):
        self.time=0
        self.time_step=pas_t
        self.hp = set(i for i in range(24*60) if 7*60 <= i%1440 < 23*60)  # set = lookup O(1)

    def tarif_hp_hc(self):
        "Attention il faut convertir en equaivalent Wh ou equaivalent batterie"
        price = 0.20 if self.time % 1440 in self.hp else 0.10
        self.time += self.time_step
        return price
    
    # def tarif_hp_hc(self):
        

    #     hp=[i for i in range(24*60) if (i%1440>=7*60 and i%1440<23*60)]
    #     if self.time%1440 in hp:
    #         self.time+=self.time_step
    #         return 0.20
    #     else: 
    #         self.time+=self.time_step           
    #         return 0.10
        
    def initialisation(self,forecast):
        self.vision=[0]*(forecast)
        for i in range(forecast):
            a=self.tarif_hp_hc()
            self.vision[i]=a
        return self.vision

    def update_vision(self):
        self.vision.pop(0)
        self.vision.append(self.tarif_hp_hc())
        return self.vision

        
        


if __name__ == "__main__":
    prix_day = PrixDay()
    prix = [prix_day.tarif_hp_hc() for i in range(24*60)]


    prix_day2=PrixDay()
    prix_day2.initialisation(5)
    for i in range(100):
        print(prix_day2.update_vision())
    print(prix_day2.vision)
    plt.plot(prix)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Price (€/kWh)")
    plt.title("Electricity Price over 24 hours")
    plt.grid()
    plt.show()