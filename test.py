import numpy as np

target_vol_pct = 10.0
tangency_vol_pct = 11.36
ratio = target_vol_pct / tangency_vol_pct
cash_weight = 1.0 - ratio
print(f"ratio: {ratio}")
print(f"cash_weight: {cash_weight}")
