"""
Environnement class definition
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.env_utils import *
from typing import Optional
from src.conso.generate_conso_day import ConsoDay
from src.conso.generate_prix import PrixDay
from src.conso.generate_prod import ProdDay
class HouseEnv(gym.Env):
    
    def __init__(self,capacity=10,forecast=10,Tmax=1000,min_price=0,max_price=100,max_prod=10,max_conso=10):
        self.cap = capacity
        self.forecast = forecast
        self.max_price = max_price
        self.tmax = Tmax
        self.max_prod = max_prod
        self.max_conso = max_conso

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                # "battery_%" : spaces.Box(0, capacity,shape=(1,),dtype=float),
                # "house_conso" : spaces.Box(0, np.inf,shape=(forecast-1,),dtype=float),
                # "solar_prod" : spaces.Box(0,np.inf,shape=(forecast-1,),dtype=float)

                "battery_%" : spaces.Discrete(capacity+1),                                # Current battery available charge
                "house_conso" : spaces.MultiDiscrete(max_conso*np.ones((forecast-1,))), # Current and foresable conso
                "solar_prod" : spaces.MultiDiscrete(max_prod*np.ones((forecast-1,))),   # Current and foresable production
                "price" : spaces.Box(min_price,max_price,(forecast-1,),dtype=float),    # Current and forseable price
                "time" : spaces.Discrete(Tmax)                                          # Current time step
            }
        )

        self.reset()

        # Rajouté par le D (ça servira plus tard tkt le chauve) - 'O' 
        @property  
        def time(self):
            return self._time
        
        @time.setter
        def time(self, value):
            self._time = value
            self._day = value // 96   # 96 pas de 15 min par jour
        @property
        def day(self):
            return self._day

    def _get_obs(self):
        obs = {
            "battery_%" : self._battery,
            "house_conso" : self._conso.vision,
            "solar_prod" : self._prod.vision,
            "price" : self._price.vision,
            "time" : self.time
        }
        return obs

    def reset(self,seed: Optional[int] = None):
        super().reset(seed=seed)
        self._battery = 0                           # Random ?
        # self._conso = np.zeros((self.forecast-1,),dtype=int)  # TO CHANGE 
        self._conso= ConsoDay() #Changements (normalement ça marche ptet un pb avec time (moi c'est juste en fonction de l'itération car necessaire en fonction de l'heure de la journee ))
        self._conso.initialisation(self.forecast)
        self._prod = ProdDay() #Changements (normalement ça marche ptet un pb avec time (moi c'est juste en fonction de l'itération car necessaire en fonction de l'heure de la journee ))
        self._prod.initialisation(self.forecast)
        self._price = PrixDay() #Changements (normalement ça marche ptet un pb avec time (moi c'est juste en fonction de l'itération car necessaire en fonction de l'heure de la journee ))
        self._price.initialisation(self.forecast)
        self.time = 0
        return self._get_obs(), {}

    def step(self,action):
        """
        Environnement reaction to an action of the agent
        Reward is proportional to the quantity of electricity
        to buy or sell 
        
        :param self: Description
        :param action: Description
        """

        assert self.action_space.contains(action)
        done = False

        # Quantity of electricity to provide at current time
        to_provide = self._conso.vision[0] - self._prod.vision[0]

        if action: # Discharge

            if to_provide > self._battery: # Need to buy electricity
                reward = - (to_provide - self._battery) * self._price.vision[0]/self.max_price
                self._battery = 0

            else: # Can sell electricity
                self._battery -= max(0,to_provide)
                reward = max(0,-1*to_provide) * self._price.vision[0]/self.max_price

        else: # Load

            battery_gap = self.cap - self._battery
            
            if to_provide > 0: # Need to buy
                reward = - (to_provide + battery_gap) * self._price.vision[0]/self.max_price
                self._battery += battery_gap

            else: # Can charge and sell excess : if excess < 0 then buy the remaining
                excess = -to_provide - battery_gap
                self._battery += battery_gap
                reward = excess * self._price.vision[0]/self.max_price

        update_conso(self._conso)
        update_prod(self._prod)
        update_price(self._price)
        self.time += 1
        if self.time >= self.tmax:
            done = True

        return self._get_obs(), reward, done, {}
    


if __name__ == "__main__":
    env= HouseEnv()
    obs, _ = env.reset()
    print(obs)
    action= 1
    obs, reward, done, _ = env.step(action)
    print(obs)
    print(reward)
    action= 0
    obs, reward, done, _ = env.step(action)
    print(obs)
    print(reward)