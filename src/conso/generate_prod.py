
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class ProdDay:
    def __init__(self,states=[0,1,2],ensoleillements=[0.9,0.5,0.1],pas_t=15,battery_unit=1):
        self.state = 0
        self.battery_unit = battery_unit
        self.states = states
        self.ensoleillements = ensoleillements
        self.pas_t=pas_t
        self.day=0
        self.time=0

        

    
    def markov_simple_init(self):
        self.prod_mu=[420 * np.exp(-0.5 * ((t - 12*60) / 120) ** 2) * ((t >= 5.75*60) & (t <= 18.25*60)) for t in np.arange(0, 60*24)]
        self.mat_transition = np.array([[0.6, 0.3, 0.1],
                                   [0.3, 0.4, 0.3],
                                   [0.1, 0.3, 0.6]])
        
    def markov_simple_update(self):
        self.state = self.states[np.random.choice(len(self.states), p=self.mat_transition[self.day % len(self.states)])]
        self.day += 1
        print(f"Day {self.day}: State {self.state}, Ensoleillement {self.ensoleillements[self.state]}")
    def iteration_batterie(self):
        if self.time>=24*60-1:
            
            self.time=0
            self.markov_simple_update()
        result = int(self.prod_mu[self.time]*self.ensoleillements[self.state]/self.battery_unit *self.pas_t/60) #attendtion on veut des equivalent Wh
        self.time+=self.pas_t
        return result
        


if __name__ == "__main__":
    prod_day = ProdDay()
    prod_day.markov_simple_init()
    prod_test= [prod_day.iteration_batterie() for t in range(0, 15*24*60, prod_day.pas_t)]
  
    plt.plot(prod_test)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Production (unité de batterie)")
    plt.title("Solar Production over 24 hours")
    plt.grid()
    plt.show()