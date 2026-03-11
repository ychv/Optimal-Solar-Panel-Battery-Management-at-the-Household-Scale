"""
Environnement class definition
"""

import gym
from gym import spaces
import numpy as np
from env_utils import *
from typing import Optional

class HouseEnv(gym.Env):
    
    def __init__(self,capacity=10,forecast=10,max_iter=100):
        self.cap = capacity
        self.forecast = forecast
        self.max_iter = max_iter

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                # "battery_%" : spaces.Box(0, capacity,shape=(1,),dtype=float),
                # "house_conso" : spaces.Box(0, np.inf,shape=(forecast-1,),dtype=float),
                # "solar_prod" : spaces.Box(0,np.inf,shape=(forecast-1,),dtype=float)
                "battery_%" : spaces.Discrete(capacity),
                "house_conso" : spaces.MultiDiscrete(capacity*np.ones((forecast-1,))),
                "solar_prod" : spaces.MultiDiscrete(capacity*np.ones((forecast-1,)))
            }
        )

        self.reset()

    def _get_obs(self):
        obs = {
            "battery_%" : self._battery,
            "house_conso" : self._conso,
            "solar_prod" : self._prod
        }
        return obs

    def reset(self,seed: Optional[int] = None):
        super().reset(seed=seed)
        self._battery = 0                           # Random ?
        self._conso = np.zeros((self.forecast-1,))  # TO CHANGE 
        self._prod = np.zeros((self.forecast-1,))   # TO CHANGE
        self._iter = 0

    def step(self,action):
        assert self.action_space.contains(action)
        if action: # Discharge
            to_provide = max(0,self._conso[0] - self._prod[0])
            if to_provide > self._battery:
                self._battery = 0
                reward = -1
            else:
                self._battery -= to_provide
                reward = 1

        else: # Load
            if self._conso[0] > self._prod[0]:
                reward = -1
            else:
                self._battery += max(self.cap,self._conso[0]-self._prod[0])
                reward = 1

        self._conso = update_conso(self._conso)
        self._prod = update_prod(self._prod)
        self._iter += 1
        if self._iter >= self.max_iter:
            done = True

        return self._get_obs(), reward, done, {}