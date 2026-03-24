"""
Configuration file for launching scripts

1. Environement configuration

2. Deep Q Learning configuration
"""

env_config = {
    'seed' : 42,
    'num_episodes' : 400,

    'capacity' : 10,
    'forecast' : 10,
    'tmax' : 1000,
    'min_price' : 0,
    'max_price' : 1,

}

DL_config = {
    'BATCH_SIZE' : 128,
    'GAMMA' : 0.99,
    'EPS_START' : 0.9,
    'EPS_END' : 0.01,
    'EPS_DECAY' : 2500,
    'TAU' : 0.005,
    'LR' : 3e-4,

    'memory' : 1000,
}