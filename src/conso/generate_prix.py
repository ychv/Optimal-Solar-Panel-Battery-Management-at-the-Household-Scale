import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class PrixDay:
    def __init__(self,pas_t=15):
        self.time=0
        self.time_step=pas_t
    
    def tarif_hp_hc(self):

        hp=[i for i in range(24*60) if (i%1440>=7*60 and i%1440<23*60)]
        if self.time%1440 in hp:
            self.time+=self.time_step
            return 0.20
        else: 
            self.time+=self.time_step           
            return 0.10
        


if __name__ == "__main__":
    prix_day = PrixDay()
    prix = [prix_day.tarif_hp_hc() for i in range(24*60)]
    plt.plot(prix)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Price (€/kWh)")
    plt.title("Electricity Price over 24 hours")
    plt.grid()
    plt.show()