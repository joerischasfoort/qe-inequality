from init_objects import *
from qe_inequality_model import *
import time

start_time = time.time()

# # 1 setup parameters
parameters = {"fundamental_values": [105, 166], "asset_types": ['stock', 'stock'],
              "trader_sample_size": 22, "n_traders": 500,
              "ticks": 1552, "std_fundamentals": [0.02, 0.053],
              "std_noise": 0.159, "w_random": 0.056,
              "strat_share_chartists": 0.41,
              "init_assets": [740, 260], "base_risk_aversion": 4.051,
              'spread_max': 0.004, "horizon": 200,
              "fundamentalist_horizon_multiplier": 2.2,
              "trades_per_tick": 3, "mutation_intensity": 0.0477,
              "average_learning_ability": 0.02, 'money_multiplier': 1.0,
              "bond_mean_reversion": 0.0, 'cb_pf_range': 0.2,
              "qe_perc_size": 0.20, "cb_size": 0.02, "qe_asset_index": 0}

seed = 0
# 2 initialise model objects
traders, central_bank, orderbook = init_objects_active_cb(parameters, seed)

# 3 simulate model
traders, central_bank, orderbook = qe_ineq_active_cb_model(traders, central_bank, orderbook, parameters, seed)

print("The simulations took", time.time() - start_time, "to run")



