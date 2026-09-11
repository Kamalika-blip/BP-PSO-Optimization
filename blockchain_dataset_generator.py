"""
blockchain_dataset_generator.py
---------------------------------
Generates a synthetic blockchain dataset with:
  - gas_price  (Gwei)    : price per unit of gas
  - gas_usage  (units)   : computational units consumed
  - latency    (seconds) : T_propagation + T_validation + T_block

Derived columns:
  - gas_total   = gas_price + gas_usage   (combined gas metric as per doc)
  - energy_cost = gas_price * gas_usage   (ETH cost proxy)
  - fitness     = gas_total + latency     (multi-objective score, lower = better)
"""

import numpy as np
import pandas as pd

# -------------------------------------------------------
# SEED FOR REPRODUCIBILITY
# -------------------------------------------------------
np.random.seed(42)

N = 500  # Number of samples

# -------------------------------------------------------
# SIMULATE REALISTIC BLOCKCHAIN PARAMETERS
# -------------------------------------------------------

# Gas Price: 1 - 100 Gwei (log-normal, clusters around 10-30 Gwei)
gas_price = np.random.lognormal(mean=np.log(20), sigma=0.6, size=N)
gas_price = np.clip(gas_price, 1.0, 100.0)

# Gas Usage: 21000 - 200000 units
# Simple transfers ~21000, smart contract calls up to 200000
gas_usage = np.random.lognormal(mean=np.log(60000), sigma=0.7, size=N)
gas_usage = np.clip(gas_usage, 21000, 200000)

# Latency components (seconds):
#   T_propagation : 0.05 - 0.50 s
#   T_validation  : 0.10 - 2.00 s
#   T_block       : 2.00 - 15.0 s (depends on network congestion)
T_propagation = np.random.uniform(0.05, 0.5,  size=N)
T_validation  = np.random.uniform(0.10, 2.0,  size=N)
T_block       = np.random.uniform(2.00, 15.0, size=N)
latency       = T_propagation + T_validation + T_block

# -------------------------------------------------------
# DERIVED COLUMNS
# -------------------------------------------------------
gas_total   = gas_price + gas_usage        # combined gas metric
energy_cost = gas_price * gas_usage        # proxy for ETH cost (Gwei * units)
fitness     = gas_total + latency          # raw multi-objective score

# -------------------------------------------------------
# BUILD DATAFRAME
# -------------------------------------------------------
df = pd.DataFrame({
    'sample_id'     : range(1, N + 1),
    'gas_price'     : np.round(gas_price,     4),
    'gas_usage'     : np.round(gas_usage,     2),
    'T_propagation' : np.round(T_propagation, 4),
    'T_validation'  : np.round(T_validation,  4),
    'T_block'       : np.round(T_block,       4),
    'latency'       : np.round(latency,       4),
    'gas_total'     : np.round(gas_total,     4),
    'energy_cost'   : np.round(energy_cost,   4),
    'fitness'       : np.round(fitness,       4),
})

# -------------------------------------------------------
# SAVE
# -------------------------------------------------------
output_file = 'blockchain_dataset.csv'
df.to_csv(output_file, index=False)

SEP = '-' * 55
print(f"[OK] Dataset generated: {N} samples -> '{output_file}'")
print(f"\n{SEP}")
print(f"{'Parameter':<20} {'Min':>10} {'Max':>10} {'Mean':>10}")
print(SEP)
for col in ['gas_price', 'gas_usage', 'latency', 'gas_total', 'energy_cost', 'fitness']:
    print(f"{col:<20} {df[col].min():>10.2f} {df[col].max():>10.2f} {df[col].mean():>10.2f}")
print(SEP)
print("\nFirst 5 rows:")
print(df[['sample_id', 'gas_price', 'gas_usage', 'latency', 'gas_total', 'fitness']].head().to_string(index=False))
