"""
Environnement class definition
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from src.conso.generate_conso_day import ConsoDay
from src.conso.generate_prix import PrixDay
from src.conso.generate_prod import ProdDay
from scripts.config import env_config

class HouseEnv(gym.Env):
    
    def __init__(self,capacity=env_config['capacity'],forecast=env_config['forecast'],Tmax=env_config['tmax'],min_price=env_config['min_price'],
                 max_price=env_config['max_price'],max_prod=env_config['max_prod'],max_conso=env_config['max_conso']):
        self.cap = capacity
        self.forecast = forecast
        self.max_price = max_price
        self.tmax = Tmax
        self.max_prod = max_prod
        self.max_conso = max_conso

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "battery_%" : spaces.Box(0,capacity,(1,),dtype=float),                  # Current battery available charge
                "house_conso" : spaces.Box(0,max_conso,(forecast-1,),dtype=float),      # Current and foresable conso
                "solar_prod" : spaces.Box(0,max_prod,(forecast-1,),dtype=float),        # Current and foresable production
                "price" : spaces.Box(min_price,max_price,(forecast-1,),dtype=float),    # Current and forseable price
                "time" : spaces.Discrete(Tmax)                                          # Current time step
            }
        )

        self.reset()

        @property  
        def time(self):
            return self._time
        
        @time.setter
        def time(self, value):
            self._time = value
            self._day = value // 96   # 96 steps of 15 min per day
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
        self._battery = 0
        self._conso= ConsoDay(mean_day=env_config['mean_day'],sigma=env_config['conso_sigma'],dt=env_config['dt'],pas_t=env_config['time_step_size'],battery_unit=env_config['capacity'])
        self._conso.initialisation(self.forecast)
        self._prod = ProdDay(states=env_config['prod_states'],ensoleillements=env_config["sunshine"],pas_t=env_config['time_step_size'],battery_unit=env_config["capacity"])
        self._prod.initialisation(self.forecast)
        self._price = PrixDay(pas_t=env_config['time_step_size'])
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

        if action == 2: # Discharge

            if to_provide > self._battery: # Need to buy electricity
                reward = - (to_provide - self._battery) * self._price.vision[0]/self.max_price
                self._battery = 0

            else: # Can sell electricity
                self._battery -= max(0,to_provide)
                reward = max(0,-1*to_provide) * self._price.vision[0]/self.max_price

        elif action == 1: # Load

            battery_gap = self.cap - self._battery
            
            if to_provide > 0: # Need to buy
                reward = - (to_provide + battery_gap) * self._price.vision[0]/self.max_price
                self._battery += battery_gap

            else: # Can charge and sell excess : if excess < 0 then buy the remaining
                excess = -to_provide - battery_gap
                self._battery += battery_gap
                reward = excess * self._price.vision[0]/self.max_price

        else: # Idle
            if to_provide > 0: # Need to buy
                reward = -to_provide * self._price.vision[0]/self.max_price
            else: # Can sell
                reward = -to_provide * self._price.vision[0]/self.max_price

        self._conso.update_vision()
        self._prod.update_vision()
        self._price.update_vision()
        self.time += 1
        if self.time >= self.tmax:
            done = True

        reward = np.clip(reward/10,-1,1)

        return self._get_obs(), reward, done, {}
    
class HouseEnvSimple(HouseEnv):

    def __init__(self, capacity=10, forecast=10, Tmax=1000, min_price=0.1, max_price=0.2, max_prod=100, max_conso=55):
        super().__init__(capacity, forecast, Tmax, min_price, max_price, max_prod, max_conso)

    def _get_obs(self):
        obs = {
            "battery_%" : self._battery,
            "house_conso" : self._conso.vision,
            "solar_prod" : self._prod.vision,
            "price" : self._price,
            "time" : self.time
        }
        return obs

    def reset(self,seed=None):
        super().reset(seed)
        self._price = [1]*self.forecast

        return self._get_obs(), {}

    def step(self,action):
        assert self.action_space.contains(action)
        done = False

        # Quantity of electricity to provide at current time
        to_provide = self._conso.vision[0] - self._prod.vision[0]

        if action == 2: # Discharge

            if to_provide > self._battery: # Need to buy electricity
                reward = - (to_provide - self._battery) * self._price[0]/self.max_price
                self._battery = 0

            else: # Can sell electricity
                self._battery -= max(0,to_provide)
                reward = max(0,-1*to_provide) * self._price[0]/self.max_price

        elif action == 1: # Load

            battery_gap = self.cap - self._battery
            
            if to_provide > 0: # Need to buy
                reward = - (to_provide + battery_gap) * self._price[0]/self.max_price
                self._battery += battery_gap

            else: # Can charge and sell excess : if excess < 0 then buy the remaining
                excess = -to_provide - battery_gap
                self._battery += battery_gap
                reward = excess * self._price[0]/self.max_price

        else: # Idle
            if to_provide > 0: # Need to buy
                reward = -to_provide * self._price[0]/self.max_price
            else: # Can sell
                reward = -to_provide * self._price[0]/self.max_price

        self._conso.update_vision()
        self._prod.update_vision()
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
    action = 2
    obs, reward, done, _ = env.step(action)
    print(obs)
    print(reward)