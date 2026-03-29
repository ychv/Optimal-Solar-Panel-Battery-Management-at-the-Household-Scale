"""
Configuration file for launching scripts

1. Environement configuration

2. Deep Q Learning configuration
"""

env_config = {
    'seed' : 42,
    'num_episodes' : 500,

    'capacity' : 100,
    'forecast' : 10,
    'tmax' : 1000,
    'min_price' : 0.1,
    'max_price' : 0.2,
    'max_prod' : 100,
    'max_conso' : 55,

    'mean_day' : None,
    'conso_sigma' : 200,
    'dt' : 0.01,
    'time_step_size' : 15,

    'prod_states' : [0,1,2],
    'sunshine' : [0.9,0.5,0.1]

}

DL_config = {
    'BATCH_SIZE' : 128,
    'GAMMA' : 0.99,
    'EPS_START' : 0.9,
    'EPS_END' : 0.01,
    'EPS_DECAY' : 30000,
    'TAU' : 0.005,
    'LR' : 3e-4,

    'memory' : 100000,
}