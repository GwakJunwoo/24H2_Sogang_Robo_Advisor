![Sogang Robo Logo](Sogang Robo Advisor/sogang-robo-logo-professional.svg)

# Roborich

**Roborich** is a Python-based portfolio optimization and backtesting engine that supports asset allocation strategies with flexible handling of assets having inconsistent time periods or missing data.

## Features

### 1. **Asset Assumption Modeling**
- Calculates expected returns and covariance matrices efficiently, even with missing data.
- Resamples data weekly and handles NaN values during calculations.

### 2. **Optimization**
- Implements Mean-Variance Optimization with risk aversion tuning.
- Ensures stability by correcting non-positive semi-definite covariance matrices.
- Handles invalid data gracefully during optimization.

### 3. **Backtesting**
- Performs robust portfolio backtesting with periodic rebalancing.
- Measures key performance metrics like:
  - **Cumulative Returns**
  - **Maximum Drawdown (MDD)**
  - **Sharpe Ratio**
- Visualizes results for better decision-making.

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
