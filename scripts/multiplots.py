import matplotlib.pyplot as plt
import pandas as pd

parent_path = './results/data/'
NUM_EP = 100

paths = [
    # ref
    "DQL_Mean_reward_500ep_100cap_10forecast_0.99gamma_decay30ep.csv",

    # Forecast test
    # "DQL_Mean_reward_500ep_100cap_1forecast.csv",
    # "DQL_Mean_reward_500ep_100cap_1000forecast.csv",

    # Capacity test
    # 'DQL_Mean_reward_100ep_10cap_10forecast.csv',
    # 'DQL_Mean_reward_500ep_50cap_10forecast.csv',
    # 'DQL_Mean_reward_100ep_1000cap_10forecast.csv',
    

    # Decay test
    # 'DQL_Mean_reward_100ep_100cap_10forecast_0.99gamma_decay1.0ep.csv',
    # 'DQL_Mean_reward_100ep_100cap_10forecast_0.99gamma_decay100.0ep.csv',

    # Gamma test
    'DQL_Mean_reward_100ep_100cap_10forecast_0.5gamma_decay30.0ep.csv',
    'DQL_Mean_reward_100ep_100cap_10forecast_0.1gamma_decay30.0ep.csv',
]

labels = [
    'Gamma = 0.99',
    'Gamma = 0.5',
    'Gamma = 0.1',
]

i=0
for path in paths:
    data = pd.read_csv(parent_path + path)
    # plt.plot(data.to_numpy()[:NUM_EP],label = path)
    plt.plot(data.to_numpy()[:NUM_EP],label = labels[i])
    i +=1

plt.xlabel('Episodes')
plt.ylabel('Mean reward per episode')
plt.legend()
plt.grid()
plt.show()