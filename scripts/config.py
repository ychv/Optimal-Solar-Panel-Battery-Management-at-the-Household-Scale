"""
Configuration file for launching scripts

1. Environement configuration

2. Deep Q Learning configuration
"""

env_config = {
    'seed' : 42,
    'num_episodes' : 100,

    'capacity' : 10,
    'forecast' : 10,
    'tmax' : 1000,
    'min_price' : 0,
    'max_price' : 1,

}

DL_config = {
    'BATCH_SIZE' : 128,
    'GAMMA' : 0.99,
    'EPS' : 0.2,
    'TAU' : 0.005,
    'LR' : 3e-4,

    'memory' : 1000,
}