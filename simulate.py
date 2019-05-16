from init_objects import *
from qe_inequality_model import *
import time

start_time = time.time()

# # 1 setup parameters
parameters = {"fundamental_values": [105, 166], "asset_types": ['bond', 'stock'],
              "trader_sample_size": 22, "n_traders": 500,
              "ticks": 500, "std_fundamentals": [0.1, 0.1],
              "std_noise": 0.1007, "w_random": 0.1,
              "strat_share_chartists": 0.8,
              "init_assets": [200, 200], "base_risk_aversion": 1.0,
              'spread_max': 0.004087, "horizon": 200,
              "fundamentalist_horizon_multiplier": 0.2,
              "trades_per_tick": 3, "mutation_intensity": 0.0,
              "average_learning_ability": 0.0, 'money_multiplier': 1.0,
              "bond_mean_reversion": 0.8,
              "qe_perc_size": 0.4, "qe_start": 2, "qe_end": 10, "qe_asset_index": 0}


seed = 4

# 2 initialise model objects
traders, central_bank, orderbook = init_objects_qe_ineq(parameters, seed)

# 3 simulate model
traders, central_bank, orderbook = qe_ineq_model(traders, central_bank, orderbook, parameters, seed)

print("The simulations took", time.time() - start_time, "to run")
