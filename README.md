![Sogang Robo Logo](sogang-robo-logo-professional.svg)

**Roborich** is a Python-based portfolio optimization and backtesting engine designed to support hierarchical asset allocation strategies and various optimization techniques. The tool provides flexibility in integrating multiple optimizers, calculating key performance metrics, and visualizing portfolio evaluations.

By combining modules for **assumption modeling**, **optimization**, and **backtesting**, Roborich enables seamless end-to-end workflows for portfolio construction and evaluation:
- Use the `AssetAssumption` module to calculate expected returns and covariance matrices from historical price data.
- Define a hierarchical asset tree with the `Tree` class and apply modular optimizers to different levels of the hierarchy.
- Execute dynamic rebalancing and evaluate portfolio performance over time with the `Backtest` engine.
- Compare strategy results with benchmarks and visualize outcomes for deeper insights.

This integrated approach simplifies portfolio management tasks, providing both flexibility and robust analytics in a single framework.


## Features

### 1. **Building Block Approach**
- **Tree-Based Hierarchical Optimization**:
  - Assets are structured hierarchically using parent-child relationships.
  - Each level of the hierarchy can apply a different optimization method.
- **Modular Optimizers**:
  - Multiple optimization techniques can be applied at different levels of the tree.
  - Supports integration of custom optimizers for specific strategies.

### 2. **Assumption Modeling**
The `AssetAssumption` class calculates:
- **Expected Returns**:
  - Simple historical expected returns based on a rolling window.
  - Expected returns using the **CAPM (Capital Asset Pricing Model)**.
- **Covariance Matrix**:
  - Asset return covariances calculated from historical data.
  - Supports rolling window calculations to focus on recent data trends.

### 3. **Supported Optimizers**
1. **Mean-Variance Optimizer**:
   - Balances risk and return using covariance matrices.
   - Requires expected returns and covariance matrix as inputs.
2. **Equal Weight Optimizer**:
   - Assigns equal weights to all assets within the group.
3. **Dynamic Risk Optimizer**:
   - Allocates weights dynamically based on risk tolerance and investment horizon.
4. **Risk Parity Optimizer**:
   - Balances risk contribution equally among assets.
5. **Goal-Based Optimizer**:
   - Focuses on achieving specific investment goals using Monte Carlo simulations.

### 4. **Backtesting**
The `Backtest` class simulates portfolio performance over a specified time frame. It integrates with the `Pipeline` class to dynamically rebalance portfolios and evaluate performance metrics.

- **Dynamic Rebalancing**:
  - Rebalances the portfolio at specified dates based on optimization outputs from the `Pipeline`.
  - Handles missing or incomplete data by forward-filling values to ensure continuity in calculations.
  - Tracks portfolio value changes over time, allowing for detailed performance evaluation.

### 5. **Performance Metrics Evaluation**
- Calculates comprehensive investment metrics, including:
  - **Cumulative Return**: Total return over the evaluation period.
  - **CAGR** (Compound Annual Growth Rate): Annualized portfolio growth rate.
  - **Sharpe Ratio**: Risk-adjusted return measurement.
  - **Sortino Ratio**: Focused risk-adjusted return using downside risk.
  - **Max Drawdown**: Largest peak-to-trough decline during the evaluation period.
  - **Annualized Volatility**: Yearly volatility of returns.
  - **Calmar Ratio**: Return-to-risk ratio using maximum drawdown.
  - **Skewness**: Asymmetry of return distribution.
  - **Kurtosis**: "Fat-tailedness" of the return distribution.
  - **Expected Daily/Monthly/Yearly Returns**: Anticipated return values over different time horizons.
  - **Kelly Criterion**: Optimal betting fraction for reinvestment.
  - **VaR (Value at Risk)**: Expected loss under adverse market conditions.
  - **CVaR (Conditional VaR)**: Expected loss beyond the VaR threshold.


## Investment Types(risk_level)

| Type               | Code |
|--------------------|------|
| 안정형              | 1    |
| 안정추구형          | 2    |
| 위험중립형          | 3    |
| 적극투자형          | 4    |
| 공격투자형          | 5    |

## Investment Goal

| Type               | Code |
|--------------------|------|
| 목돈 마련              | 1    |
| 결혼자금 준비          | 2    |
| 노후자금 준비          | 3    |
| 장기수익 창출          | 4    |

## Installation and Setup

1. **Install Dependencies**:
   Run the following to install required libraries:
   ```bash
   pip install numpy pandas matplotlib cvxpy tqdm

2. **Usage Example**:
   Run the following Example:
   ```bash
   main(codes=['069500','139260','161510','273130','439870','251340','114260'], risk_level=5, investor_goal=4)
