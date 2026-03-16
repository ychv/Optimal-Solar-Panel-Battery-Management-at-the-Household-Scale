from src.env import HouseEnv

env = HouseEnv()
action_list = [1,0,1]
for action in action_list:
    step_results = env.step(action)
    state = step_results[0]
    reward = step_results[1]
    done = step_results[2]
    print(state,reward,done)