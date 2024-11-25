
from typing import List, Optional, Tuple, Union, Callable
import numpy as np
import cvxpy as cp
import collections
from Tree import *
from BaseOptimizer import *

def portfolio_variance(weights, covariance_matrix):
    return cp.quad_form(weights, cp.Constant(covariance_matrix))

def mean_return(weights, expected_returns):
    return cp.matmul(weights, expected_returns)

def is_positive_semidefinite(matrix: np.ndarray) -> bool:
    """Check if a matrix is positive semi-definite (PSD) by verifying all eigenvalues are non-negative."""
    return np.all(np.linalg.eigvals(matrix) >= 0)

def make_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    """Make a matrix positive semi-definite by adjusting its eigenvalues if necessary."""
    min_eigenvalue = np.min(np.linalg.eigvals(matrix))
    if min_eigenvalue < 0:
        matrix += np.eye(matrix.shape[0]) * (-min_eigenvalue + 1e-6)
    return matrix

def mean_variance_optimizer(
    nodes: List[Any],  # Assuming Node has 'name' attribute
    covariance_matrix: np.ndarray,
    expected_returns: np.ndarray,
    weight_bounds: Union[List[Tuple], Tuple] = (0, 1),
    risk_aversion: float = 0.5
) -> List[float]:

    def mean_variance_objective(w, cov_matrix, exp_returns, risk_aversion):
        return risk_aversion * portfolio_variance(w, cov_matrix) - mean_return(w, exp_returns)

    n_assets = len(nodes)
    tickers = [node.name for node in nodes]

    if covariance_matrix.shape[0] != n_assets or covariance_matrix.shape[1] != n_assets:
        raise ValueError(f"Covariance matrix dimensions {covariance_matrix.shape} do not match the number of assets ({n_assets}).")

    # 공분산 행렬이 양의 정부호인지 확인하고, 아니라면 수정합니다.
    if not is_positive_semidefinite(covariance_matrix):
        covariance_matrix = make_positive_semidefinite(covariance_matrix)

    if n_assets == 1:
        return [1.0]
    optimizer = BaseConvexOptimizer(n_assets, tickers=tickers, weight_bounds=weight_bounds)
    optimizer.convex_objective(
        lambda w: mean_variance_objective(w, covariance_matrix, expected_returns, risk_aversion),
        weights_sum_to_one=True
    )

    return list(optimizer.clean_weights().values())

def equal_weight_optimizer(
    nodes: List[Any],  # Assuming Node has 'name' attribute
    weight_bounds: Union[List[Tuple], Tuple] = (0, 1)
) -> List[float]:
    n = len(nodes)
    if n == 0:
        return []
    return [1.0 / n] * n

def dynamic_risk_optimizer(
    nodes: List[Any],
    covariance_matrix: np.ndarray,
    risk_tolerance: float = 0.5,
    goal_period: int = 10,
    weight_bounds: Union[List[Tuple], Tuple] = (0, 1)
) -> List[float]:
    """
    Optimizer based on dynamic risk allocation.
    Args:
        nodes: List of assets as nodes.
        covariance_matrix: Covariance matrix of asset returns.
        risk_tolerance: Risk tolerance level (1-10).
        goal_period: Target investment horizon (e.g., years).

    Returns:
        List of portfolio weights.
    """
    n_assets = len(nodes)
    if n_assets == 0:
        return []

    # Risk-adjusted volatility
    stability_factor = goal_period / 10
    adjusted_volatility = np.diag(covariance_matrix) ** (risk_tolerance / stability_factor)

    # Inverse volatility weights
    inverse_volatility = 1 / adjusted_volatility
    weights = inverse_volatility / np.sum(inverse_volatility)

    return list(weights)

def risk_parity_optimizer(
    nodes: List[Any],
    covariance_matrix: np.ndarray,
    risk_aversion: float = 0.5,
    weight_bounds: Union[List[Tuple], Tuple] = (0, 1)
) -> List[float]:
    """
    Optimizer based on risk parity allocation.
    Args:
        nodes: List of assets as nodes.
        covariance_matrix: Covariance matrix of asset returns.
        risk_aversion: Risk aversion parameter (1: neutral, >1: less risk-tolerant).

    Returns:
        List of portfolio weights.
    """
    n_assets = len(nodes)
    if n_assets == 0:
        return []

    # Risk-adjusted volatility
    volatilities = np.sqrt(np.diag(covariance_matrix))
    adjusted_volatility = volatilities ** risk_aversion
    inverse_volatility = 1 / adjusted_volatility
    weights = inverse_volatility / np.sum(inverse_volatility)

    return list(weights)

def goal_based_optimizer(
    nodes: List[Any],  # Assuming Node has 'name' attribute
    covariance_matrix: np.ndarray,
    expected_returns: np.ndarray,
    weight_bounds: Union[List[Tuple], Tuple] = (0, 1),
    risk_aversion: float = 0.5,
    goal_amount: float = 1000000,
    goal_period: int = 10,
    simulations: int = 1000,
) -> List[float]:
    """
    Optimizer for Goal-Based Investing (GBI) using BaseConvexOptimizer.
    Args:
        nodes: List of assets as nodes.
        covariance_matrix: Covariance matrix of asset returns.
        expected_returns: Expected returns of assets.
        weight_bounds: Bounds for weights (default: (0, 1)).
        risk_aversion: Risk aversion parameter.
        goal_amount: Target investment amount.
        goal_period: Investment horizon (in years).
        simulations: Number of Monte Carlo simulations.

    Returns:
        List of portfolio weights.
    """
    n_assets = len(nodes)
    tickers = [node.name for node in nodes]

    if covariance_matrix.shape[0] != n_assets or covariance_matrix.shape[1] != n_assets:
        raise ValueError(f"Covariance matrix dimensions {covariance_matrix.shape} do not match the number of assets ({n_assets}).")

    # Check if covariance matrix is positive semi-definite
    if not is_positive_semidefinite(covariance_matrix):
        covariance_matrix = make_positive_semidefinite(covariance_matrix)

    if n_assets == 1:
        return [1.0]

    # Monte Carlo simulation to calculate success probability
    np.random.seed(42)
    ending_values = []
    for _ in range(simulations):
        weights = np.random.dirichlet(np.ones(n_assets), size=1).flatten()
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        yearly_returns = np.random.normal(portfolio_return, portfolio_volatility, goal_period)
        ending_value = np.prod(1 + yearly_returns) * goal_amount
        ending_values.append(ending_value)

    failure_prob = np.mean(np.array(ending_values) < goal_amount)
    success_prob = 1 - failure_prob

    # Adjust risk aversion based on success probability
    adjusted_risk_aversion = risk_aversion * (1 + success_prob)

    # Define GBI objective
    def gbi_objective(w, cov_matrix, exp_returns, adjusted_risk_aversion):
        portfolio_var = portfolio_variance(w, cov_matrix)
        portfolio_ret = mean_return(w, exp_returns)
        return adjusted_risk_aversion * portfolio_var - portfolio_ret

    # Use BaseConvexOptimizer for optimization
    optimizer = BaseConvexOptimizer(n_assets, tickers=tickers, weight_bounds=weight_bounds)
    optimizer.convex_objective(
        lambda w: gbi_objective(w, covariance_matrix, expected_returns, adjusted_risk_aversion),
        weights_sum_to_one=True
    )

    return list(optimizer.clean_weights().values())
