from init_objects import *
from qe_inequality_model import *
import time

start_time = time.time()

# # 1 setup parameters
parameters = {"fundamental_values": [105, 166], "asset_types": ['bond', 'stock'],
              "trader_sample_size": 10, "n_traders": 500,
              "ticks": 400, "std_fundamentals": [0.05, 0.01],
              "std_noise": 0.10696588473846724, "w_random": 0.1,
              "strat_share_chartists": 0.5,
              "init_assets": [50, 50], "base_risk_aversion": 1.0,
              'spread_max': 0.004087, "horizon": 200,
              "fundamentalist_horizon_multiplier": 0.2,
              "trades_per_tick": 3, "mutation_intensity": 0.05,
              "average_learning_ability": 0.5,
              "bond_mean_reversion": -0.001,
              "qe_perc_size": 0.1, "qe_start": 100, "qe_end": 110, "qe_asset_index": 0}

# 2 initialise model objects
traders, central_bank, orderbook = init_objects_qe_ineq(parameters, seed=0)

# 3 simulate model
traders, central_bank, orderbook = qe_ineq_model(traders, central_bank, orderbook, parameters, seed=0)

print("The simulations took", time.time() - start_time, "to run")