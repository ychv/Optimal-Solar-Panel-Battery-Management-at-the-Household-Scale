"""
Environnement class definition
"""

import gym
from gym import spaces
import numpy as np
from src.env_utils import *
from typing import Optional

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

                "battery_%" : spaces.Discrete(capacity),                                # Current battery available charge
                "house_conso" : spaces.MultiDiscrete(max_conso*np.ones((forecast-1,))), # Current and foresable conso
                "solar_prod" : spaces.MultiDiscrete(max_prod*np.ones((forecast-1,))),   # Current and foresable production
                "price" : spaces.Box(min_price,max_price,(forecast-1,),dtype=float),    # Current and forseable price
                "time" : spaces.Discrete(Tmax)                                          # Current time step
            }
        )

        self.reset()

    def _get_obs(self):
        obs = {
            "battery_%" : self._battery,
            "house_conso" : self._conso,
            "solar_prod" : self._prod,
            "price" : self._price,
            "time" : self._time
        }
        return obs

    def reset(self,seed: Optional[int] = None):
        super().reset(seed=seed)
        self._battery = 0                           # Random ?
        self._conso = np.zeros((self.forecast-1,),dtype=int)  # TO CHANGE 
        self._prod = np.zeros((self.forecast-1,),dtype=int)   # TO CHANGE
        self._price = np.zeros((self.forecast-1,),dtype=float)   # TO CHANGE
        self._time = 0

    def step(self,action):
        assert self.action_space.contains(action)
        done = False
        to_provide = self._conso[0] - self._prod[0]
        if action: # Discharge
            if to_provide > self._battery: # Need to buy electricity
                self._battery = 0
                reward = -1 * (to_provide - self._battery)* self._price[0]/self.max_price
            else: # Can sell electricity
                self._battery -= max(0,to_provide)
                reward = 1 * max(0,-1*to_provide) * self._price[0]/self.max_price

        else: # Load
            if to_provide > 0: # Need to buy
                reward = -1 * (to_provide - self._battery)* self._price[0]/self.max_price
            else: # Can charge and sell excess
                excess = -1*to_provide - (self.cap - self._battery)
                self._battery += (self.cap - self._battery)
                reward = 1 * max(0,excess) * self._price[0]/self.max_price

        self._conso = update_conso(self._conso)
        self._prod = update_prod(self._prod)
        self._price = update_price(self._price)
        self._time += 1
        if self._time >= self.tmax:
            done = True

        return self._get_obs(), reward, done, {}