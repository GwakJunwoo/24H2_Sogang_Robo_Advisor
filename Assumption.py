import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class AssetAssumption:
    def __init__(self, returns_window: int = 52, covariance_window: int = 52):
        self.returns_window = returns_window
        self.covariance_window = covariance_window

    def calculate_expected_return(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculates the expected return from weekly price data."""
        # Resample to weekly data and calculate weekly returns
        weekly_prices = price_data.resample('W').last()
        weekly_returns = weekly_prices.pct_change().astype(np.float32)

        # Convert weekly returns to numpy array
        weekly_returns = weekly_returns.to_numpy()

        if weekly_returns.shape[0] == 0 or weekly_returns.shape[0] < self.returns_window:
            # 빈 배열이거나 데이터가 부족한 경우 처리
            if np.all(np.isnan(weekly_returns)):
                # 모든 값이 NaN인 경우 -99999 반환
                expected_returns = np.full(price_data.shape[1], -99999)
            else:
                # NaN을 무시하고 평균 계산 (annualize)
                expected_returns = np.nanmean(weekly_returns, axis=0)

            # 여전히 NaN이 남아 있는 경우 -99999로 대체
            expected_returns = np.where(np.isnan(expected_returns), -99999, expected_returns)
            return pd.Series(expected_returns, index=price_data.columns)

        # Compute rolling mean for expected returns using sliding window
        rolling_means = np.lib.stride_tricks.sliding_window_view(
            weekly_returns, self.returns_window, axis=0
        ).mean(axis=2)
        expected_returns = rolling_means[-1] # Scale for annualized return

        # Handle columns with insufficient data
        valid_data_counts = np.sum(~np.isnan(weekly_returns), axis=0)
        fallback_means = []
        for col_idx in range(weekly_returns.shape[1]):
            if valid_data_counts[col_idx] == 0:
                fallback_means.append(-99999)  # If all values are NaN, set fallback to -99999
            else:
                fallback_means.append(np.nanmean(weekly_returns[:, col_idx]))
        fallback_means = np.array(fallback_means)

        # Combine results
        final_returns = np.where(valid_data_counts >= self.returns_window, expected_returns, fallback_means)
        final_returns = np.nan_to_num(final_returns, nan=-99999)  # Replace remaining NaN with -99999

        return pd.Series(final_returns, index=price_data.columns)

    def calculate_capm_expected_return(self, price_data: pd.DataFrame, risk_free_rate: float = 0.01) -> pd.Series:
        """Calculates expected return using the CAPM model."""
        weekly_prices = price_data.resample('W').last()
        weekly_returns = weekly_prices.pct_change().astype(np.float32)

        market_returns = price_data.iloc[:,0].resample('W').last().pct_change().astype(np.float32)

        betas = []
        for col in weekly_returns.columns:
            valid_data = weekly_returns[col].dropna()
            valid_market = market_returns.loc[valid_data.index].dropna()

            if len(valid_market) < 2:
                betas.append(-99999)  # Insufficient data
                continue

            beta = np.cov(valid_data, valid_market)[0, 1] / np.var(valid_market)
            betas.append(beta)

        betas = np.array(betas)
        capm_returns = risk_free_rate + betas * (market_returns.mean() - risk_free_rate)

        capm_returns = np.nan_to_num(capm_returns, nan=-99999)  # Replace NaN with fallback value
        return pd.Series(capm_returns, index=price_data.columns)


    def calculate_covariance(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculates the covariance matrix from weekly price data."""
        # Resample to weekly data and calculate weekly returns
        weekly_prices = price_data.resample('W').last()
        weekly_returns = weekly_prices.pct_change().astype(np.float32)

        # Convert weekly returns to numpy array
        weekly_returns = weekly_returns.to_numpy()

        num_assets = weekly_returns.shape[1]
        covariance_matrix = np.zeros((num_assets, num_assets))  # Initialize covariance matrix

        for i in range(num_assets):
            for j in range(i, num_assets):  # Only iterate for upper triangle (including diagonal)
                valid_data = weekly_returns[:, [i, j]]
                valid_data = valid_data[~np.isnan(valid_data).any(axis=1)]  # Remove NaN rows
                if len(valid_data) >= self.covariance_window:
                    # Calculate rolling covariance
                    cov_value = np.cov(valid_data[-self.covariance_window:], rowvar=False)[0, 1]
                elif len(valid_data) > 1:
                    # Fallback to all available data
                    cov_value = np.cov(valid_data, rowvar=False)[0, 1]
                else:
                    cov_value = 0  # No sufficient data
                covariance_matrix[i, j] = cov_value
                if i != j:
                    covariance_matrix[j, i] = cov_value  # Mirror the value to the lower triangle

        return pd.DataFrame(covariance_matrix, index=price_data.columns, columns=price_data.columns)
