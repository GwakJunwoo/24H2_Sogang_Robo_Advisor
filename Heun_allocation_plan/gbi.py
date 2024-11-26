import numpy as np
from scipy.optimize import minimize

def gbi_allocation(goals, data, risk_aversion, simulations=10000):
    final_allocations = {}
    np.random.seed(42)

    for goal in goals:
        assets = goal['assets']
        goal_amount = goal['amount']
        block_data = data[assets].pct_change().dropna()

        expected_returns = block_data.mean().values
        covariance_matrix = block_data.cov().values

        # 몬테카를로 시뮬레이션
        num_assets = len(assets)
        ending_values = []
        for _ in range(simulations):
            weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            yearly_returns = np.random.normal(portfolio_return, portfolio_volatility, goal['period'])
            ending_value = np.prod(1 + yearly_returns) * goal_amount
            ending_values.append(ending_value)

        failure_prob = np.mean(np.array(ending_values) < goal_amount)
        success_prob = 1 - failure_prob

        # 위험 선호도 조정
        adjusted_risk = risk_aversion * (1 + success_prob)

        # MVO
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.05, 0.7) for _ in range(num_assets)]

        result = minimize(
            portfolio_volatility,
            num_assets * [1. / num_assets],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            raise ValueError(f"Optimization failed for goal '{goal['name']}': {result.message}")

        weights = result.x
        final_allocations[goal['name']] = {
            asset: round(goal_amount * weight, 4) for asset, weight in zip(assets, weights)
        }

    return final_allocations

