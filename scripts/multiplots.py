import matplotlib.pyplot as plt
import pandas as pd

parent_path = './results/data/'

paths = [
    "DQL_Mean_reward_500ep_100cap.csv",
    "DQL_Mean_reward_500ep_100cap_1forecast.csv"
]

for path in paths:
    data = pd.read_csv(parent_path + path)
    plt.plot(data.to_numpy(),label = path)

plt.xlabel('Episodes')
plt.ylabel('Mean reward per episode')
plt.legend()
plt.grid()
plt.show()