
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


## Installation and Setup

1. **Install Dependencies**:
   Run the following to install required libraries:
   ```bash
   pip install numpy pandas matplotlib cvxpy


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
