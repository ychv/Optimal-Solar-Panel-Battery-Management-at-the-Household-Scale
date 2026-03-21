from src.env import HouseEnv
from tqdm import tqdm as tqdm
import random
import matplotlib.pyplot as plt

num_episode = 100 ; Tmax = 1000
seed = 42
random.seed(seed)
env = HouseEnv(max_price=1,Tmax=Tmax)

def random_policy():
    return random.randint(0,1)

rewards = []

for i in tqdm(range(num_episode)):
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