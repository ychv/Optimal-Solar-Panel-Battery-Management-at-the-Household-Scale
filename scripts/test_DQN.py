"""
Code largely from PyTorch RL tutorial : 
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

"""

from scripts.config import env_config, DL_config

import matplotlib.pyplot as plt
from itertools import count
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from src.env import HouseEnv
from src.deep_agent import DQN, ReplayMemory, select_action, optimize_model, to_features

env = HouseEnv(capacity=env_config["capacity"],
               forecast=env_config["forecast"],
               Tmax=env_config['tmax'],
               min_price=env_config['min_price'],
               max_price=env_config["max_price"])

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


# To ensure reproducibility during training, you can fix the random seeds
# by uncommenting the lines below. This makes the results consistent across
# runs, which is helpful for debugging or comparing different approaches.
#
# That said, allowing randomness can be beneficial in practice, as it lets
# the model explore different training trajectories.


seed = env_config['seed']
random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS value of epsilon
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = DL_config['BATCH_SIZE']
GAMMA = DL_config['GAMMA']
EPS_START = DL_config['EPS_START']
EPS_END = DL_config['EPS_END']
EPS_DECAY = DL_config['EPS_DECAY']
TAU = DL_config['TAU']
LR = DL_config['LR']


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(to_features(state))

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(DL_config['memory'])


steps_done = 0

episode_durations = []

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = env_config['num_episodes']
else:
    num_episodes = int(env_config['num_episodes']/12)

rewards = []

for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(to_features(state), dtype=torch.float32, device=device).unsqueeze(0)
    rwd = []
    for t in count():
        action = select_action(state,EPS_START,EPS_END,EPS_DECAY,policy_net,env,device)
        observation, reward, terminated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        rwd.append(reward)
        done = terminated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(to_features(observation), dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model(memory,BATCH_SIZE,device,policy_net,target_net,GAMMA,optimizer)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break
    
    rewards.append(torch.mean(torch.tensor(rwd)))

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel("Mean reward")
plt.grid()
plt.show()
