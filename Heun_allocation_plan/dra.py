import numpy as np
import pandas as pd

def dynamic_risk_allocation(data, risk_tolerance, goal_period):
    returns = data.pct_change().dropna()
    volatilities = returns.std()

    stability_factor = goal_period / 10  # 10년 기준 안정성 보정
    adjusted_volatility = volatilities ** (risk_tolerance / stability_factor)

    inverse_volatility = 1 / adjusted_volatility
    weights = inverse_volatility / np.sum(inverse_volatility)
    return weights.to_dict()
