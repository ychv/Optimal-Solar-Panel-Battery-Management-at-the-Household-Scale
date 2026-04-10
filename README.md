# Optimal Solar Panel Battery Management at the Houshold Scale

As solar panels and home batteries become more widely available, the question of how to best manage energy storage is becoming increasingly important. A household equipped with these systems must constantly decide: should it charge the battery or feed its excess production back into the grid? The challenge stems from uncertainty regarding consumption, solar production, and electricity prices. We propose a realistic simulation environment using stochastic differential equations coupled with a deep reinforcement learning algorithm to automatically learn an optimal management strategy.

### Repository structure

The user is expected to use ```main.py``` to launch scripts and ```scripts/config.py``` to change parameters

```
/-- results # figures of experiments (csv files are not kept)
/-- scripts
    +-- config.py           # config file to change env and agent parameters
    +-- multiplots.py       # to plot several experiments
    +-- test_DQN.py         # implementation of DQL
    +-- test_random_policy  # random policy agent
/-- src
    /-- conso               # environements models (consumption, price and solar production)
    +-- deep_agent.py       # implementation of DQL agent
    +-- env.py              # implementation of environment
+-- main.py                 # use this to launch scripts
+-- requirements.txt        # caution with torch + cuda install
```