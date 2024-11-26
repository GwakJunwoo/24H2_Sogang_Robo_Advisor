# risk_parity.py
import numpy as np
import pandas as pd

def risk_parity_allocation(data, risk_aversion):
    returns = data.pct_change().dropna()
    volatilities = returns.std() 

    # 위험 회피도 반영 (1 -> 더 높은 가중치 부여, 5 -> 변동성 덜 강조)
    adjusted_volatility = volatilities ** risk_aversion
    inverse_volatility = 1 / adjusted_volatility
    weights = inverse_volatility / np.sum(inverse_volatility)
    return weights.to_dict()
