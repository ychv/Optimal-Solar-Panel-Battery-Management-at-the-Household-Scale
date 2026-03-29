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
import pandas as pd

import torch
import torch.optim as optim

from src.env import HouseEnv, HouseEnvSimple
from src.deep_agent import DQN, ReplayMemory,  to_features, select_action, optimize_model

env = HouseEnv()

# env = HouseEnvSimple(capacity=env_config["capacity"],
#                forecast=env_config["forecast"],
#                Tmax=env_config['tmax'],
#                min_price=env_config['min_price'],
#                max_price=env_config["max_price"],
#                max_prod=env_config['max_prod'],
#                max_conso=env_config['max_conso'])

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
n_observations = len(to_features(state,env))

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(DL_config['memory'])

global steps_done
steps_done = 0

episode_durations = []

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = env_config['num_episodes']
else:
    num_episodes = int(env_config['num_episodes']/12)

rewards = [] ; actions_tot = []

for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and get its state
    actions = []
    state, info = env.reset()
    state = torch.tensor(to_features(state,env), dtype=torch.float32, device=device).unsqueeze(0)
    rwd = []
    for t in count():
        action = select_action(state,EPS_START,EPS_END,EPS_DECAY,policy_net,env,device,steps_done)
        actions.append(action.item())
        steps_done += 1
        observation, reward, terminated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        rwd.append(reward)
        done = terminated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(to_features(observation,env), dtype=torch.float32, device=device).unsqueeze(0)

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
    
    rewards.append(torch.mean(torch.tensor(rwd)).item())
    actions_tot.append(actions)

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel("Mean reward")
plt.grid()
plt.show()

actions_plot = np.array(actions_tot[-1:][:24*4]).flatten()
plt.scatter(range(len(actions_plot)),actions_plot)
plt.title('Actions used on the last 3 episodes')
plt.ylabel('Action used')
plt.grid()
plt.show()

pd.DataFrame(rewards,columns=['mean_reward']).to_csv(f'./results/DQL_Mean_reward_{env_config['num_episodes']}ep_{env_config['capacity']}cap.csv',sep=',',index=False,header=False)
pd.DataFrame(np.array(actions_tot).flatten()).to_csv(f'./results/DQL_All_actions_{env_config['num_episodes']}ep_{env_config['capacity']}cap_{env_config['time_step_size']}stepsize.csv',sep=',',index=False,header=False)