
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# DIR= os.path.dirname(os.path.abspath(__file__))
# file = os.path.join(DIR, "consceaux.txt")
# data= pd.read_csv(file, sep=";", header=5, skip_blank_lines=True)

# print(data.head())


class ConsoDay:
    def __init__(self, mean_day=None,sigma=200,dt=0.01,pas_t=15,battery_unit=10):
        if mean_day is None:
            DIR= os.path.dirname(os.path.abspath(__file__))
            file = os.path.join(DIR, "consceaux.txt")
            data= pd.read_csv(file, sep=";", header=5, skip_blank_lines=True)
            self.mean_day = data["puissance_W"].values
        else: self.mean_day = mean_day
        self.sigma = sigma
        self.dt = dt
        self.conso= [self.mean_day[0]]
        self.iteration=0
        self.pas_t=pas_t
        self.battery_unit=battery_unit
    
    def drift_simple(self,i):
        return 3*(self.mean_day[i] - self.conso[i])
    def drift(self,i):
        mu_prime = (self.mean_day[i+1]-self.mean_day[i]) / self.dt
        return mu_prime - 3*(self.conso[i]-self.mean_day[i])
    
    def update_conso(self):
        i=self.iteration
        self.iteration+=1
        self.conso.append(self.conso[i] + self.drift_simple(i)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.normal())

    def translate_result(self):
        self.conso_translated = np.array([int(self.conso[i]/self.battery_unit) for i in range(0,len(self.conso),int(self.pas_t))])
        return self.conso_translated*self.pas_t/60

    def iteration_batterie(self):
        for i in range(self.pas_t):
            self.update_conso()
           

        return int(self.conso[self.iteration-1]/self.battery_unit*self.pas_t/60) #attendtion on veut des equivalent Wh
    

if __name__ == "__main__":
    for j in range(10):
        # mean_day = data["puissance_W"].values
        # mean_day=np.concatenate((mean_day,mean_day))  # dupliquer pour faire 48h
        # mean_day=np.concatenate((mean_day,mean_day))
        sigma = 200
        dt =1./60  # 1 minute
       

        conso_day = ConsoDay(sigma=sigma,dt=dt)

        for i in range(1440):
            conso_day.update_conso()
               


    plt.plot(conso_day.translate_result())
    # print(len(conso_day.conso_translated))
    A=[conso_day.conso[0]/20]
    conso_day2 = ConsoDay(sigma=sigma,dt=dt)
    for i in range(96):
        A.append(conso_day2.iteration_batterie())

    plt.plot(A)

    plt.legend(["Conso simulée","Conso moyenne"])
    plt.show()




  