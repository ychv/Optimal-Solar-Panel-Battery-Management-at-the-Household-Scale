"""
Simple test of a random policy
"""
from scripts.config import env_config

from src.env import HouseEnv
from tqdm import tqdm as tqdm
import random
import pandas as pd
import matplotlib.pyplot as plt

num_episodes = env_config['num_episodes'] ; Tmax = env_config['tmax']
seed = env_config['seed']
random.seed(seed)
env = HouseEnv(capacity=env_config["capacity"],
               forecast=env_config["forecast"],
               Tmax=Tmax,
               min_price=env_config['min_price'],
               max_price=env_config["max_price"],
               max_prod=env_config['max_prod'],
               max_conso=env_config['max_conso'])

def random_policy():
    return env.action_space.sample()

rewards = []

for i in tqdm(range(num_episodes)):
    reward_ep = 0
    state,_ = env.reset(seed=seed)
    action = random_policy()
    new_state, reward, done, _ = env.step(action)
    reward_ep += reward
    
    while not done:
        action = random_policy()
        new_state, reward, done, _ = env.step(action)
        reward_ep += reward

    rewards.append(reward_ep/Tmax)

plt.plot(rewards)
plt.title('Random policy')
plt.xlabel('Episode')
plt.ylabel('Mean reward')
plt.grid()
plt.show()

pd.DataFrame(rewards,columns=['mean_reward']).to_csv(f'./results/Rd_Mean_reward_{env_config['num_episodes']}ep_{env_config['capacity']}cap.csv',sep=',',index=False,header=False)