# mvo.py
import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Union

def mvo_optimizer(
    nodes: List[str],
    covariance_matrix: np.ndarray,
    expected_returns: np.ndarray,
    weight_bounds: Union[List[Tuple[float, float]], Tuple[float, float]] = (0, 1),
    risk_aversion: float = 1.0
) -> List[float]:
    def mvo_objective(weights, cov_matrix, exp_returns, risk_aversion):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_return = np.dot(weights, exp_returns)
        return (risk_aversion * portfolio_variance) - portfolio_return

    n_assets = len(nodes)

    if covariance_matrix.shape[0] != n_assets or covariance_matrix.shape[1] != n_assets:
        raise ValueError(f"Covariance matrix dimensions {covariance_matrix.shape} do not match the number of assets ({n_assets}).")

    if n_assets == 1:
        return [1.0]
    
    initial_weights = np.array([1 / n_assets] * n_assets)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    bounds = weight_bounds if isinstance(weight_bounds, list) else [weight_bounds] * n_assets

    # 최적화
    result = minimize(
        fun=mvo_objective,
        x0=initial_weights,
        args=(covariance_matrix, expected_returns, risk_aversion),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    # 최적 가중치
    return list(result.x)
