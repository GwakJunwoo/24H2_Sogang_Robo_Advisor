import matplotlib.pyplot as plt
from Evaluation import *
from tqdm import tqdm
import pandas as pd
import numpy as np

class Backtest:
    def __init__(self, pipeline, price_data, rebalance_dates, trading_days):
        """
        백테스트를 초기화합니다.
        
        pipeline: Pipeline 클래스의 인스턴스
        price_data: 가격 데이터 (DataFrame, 인덱스는 날짜, 컬럼은 자산명)
        rebalance_dates: 리밸런싱 날짜 리스트 (리밸런싱은 해당 날짜까지의 데이터로 수행)
        trading_days: 거래 가능한 날짜 리스트 (리밸런싱 날짜가 휴일일 경우 가장 가까운 거래일로 조정)
        """
        self.pipeline = pipeline
        self.price_data = price_data.ffill()  # 결측값 처리
        self.rebalance_dates = rebalance_dates
        self.trading_days = trading_days
        self.allocations = []  # 리밸런싱 시점별 포트폴리오 비중 저장
        self.portfolio_values = pd.DataFrame(index=price_data.loc[str(rebalance_dates[0]):].index, columns=["Portfolio Value"])

    def get_previous_trading_day(self, date):
        """ 주어진 날짜에 가장 가까운 거래일을 반환 """
        return max([d for d in self.trading_days if d <= date])
    
    def handle_missing_data(self, data, current_date):
        """ 결측값을 이전 거래일 값으로 대체 """
        return data.ffill().loc[:current_date]  # 결측값을 이전 값으로 채우고 현재 날짜까지의 데이터를 반환
    
    def rebalance(self, current_date):
        """ 리밸런싱을 수행하고 포트폴리오 비중을 반환 """
        prev_trading_day = self.get_previous_trading_day(current_date)
        price_data_until_now = self.price_data.loc[:prev_trading_day]
        
        # 결측값 처리 (이전 데이터로 채우기)
        clean_price_data = self.handle_missing_data(price_data_until_now, prev_trading_day)
        
        # 해당 시점까지의 가격 데이터로 Assumption을 계산
        allocation = self.pipeline.run(clean_price_data)
        
        # Tree에서 리프 노드를 가져옴
        leaf_nodes = self.pipeline.universe.get_leaf_nodes()
        
        # allocation 딕셔너리의 순서를 유지한 채 리프 노드만 남김
        final_allocation = {k: v for k, v in allocation.items() if k in leaf_nodes}
        return final_allocation
    
    def run_backtest(self, initial_value=1000000):
        """ 백테스트를 실행 """
        current_value = initial_value
        for i, date in tqdm(enumerate(self.rebalance_dates)):
            allocation = self.rebalance(date)

            self.allocations.append((date, allocation))

            # 해당 리밸런싱 시점 이후부터 다음 리밸런싱까지 포트폴리오 수익률 계산
            if i < len(self.rebalance_dates) - 1:
                next_rebalance_date = self.rebalance_dates[i + 1]
            else:
                next_rebalance_date = self.price_data.index[-1]

            # 자산별 수익률 계산 및 포트폴리오 가치 업데이트
            asset_returns = self.price_data.loc[date:next_rebalance_date].pct_change()
            for day in asset_returns.index:
                # 비중이 0인 자산을 제외하고 수익률 계산
                valid_allocation = {k: v for k, v in allocation.items() if v > 0}
                valid_asset_returns = asset_returns.loc[day][valid_allocation.keys()]

                # 수익률이 NaN인 자산을 제외하고 계산
                valid_asset_returns = valid_asset_returns.dropna()
                valid_allocation_series = pd.Series(valid_allocation).loc[valid_asset_returns.index]

                # 자산별 수익률과 비중을 곱해서 일일 수익률 계산
                day_return = sum(valid_asset_returns * valid_allocation_series)
                current_value *= (1 + day_return)
                self.portfolio_values.loc[day] = current_value
        self.portfolio_values.iloc[0] = initial_value


    def calculate_performance(self):
        """ 백테스트 성과를 계산 (누적 수익률, MDD, Sharpe Ratio) """
        # 누적 수익률 계산
        cumulative_return = self.portfolio_values["Portfolio Value"].iloc[-1] / self.portfolio_values["Portfolio Value"].iloc[0] - 1
        
        # MDD(Maximum Drawdown) 계산
        running_max = self.portfolio_values["Portfolio Value"].cummax()
        drawdown = (self.portfolio_values["Portfolio Value"] - running_max) / running_max
        mdd = drawdown.min()
        
        # Sharpe Ratio 계산 (무위험 수익률 0 가정)
        daily_return = self.portfolio_values["Portfolio Value"].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_return.mean() / daily_return.std()
        
        return cumulative_return, mdd, sharpe_ratio
    
    def evaluation(self, allocation_dict):
        model = self.portfolio_values['Portfolio Value'].astype(np.int64)/100
        bench = self.price_data[['069500']].squeeze()  # DataFrame -> Series 변환

        model = model.pct_change().dropna()
        bench = bench.pct_change().dropna()

        # 공통된 인덱스 찾기
        common_index = model.index.intersection(bench.index)

        # 공통 인덱스로 데이터 정렬
        model_filtered = model.loc[common_index]
        bench_filtered = bench.loc[common_index]

        # DataFrame 생성
        evaluation_target = pd.DataFrame({
            'Benchmark': bench_filtered,
            'Model': model_filtered
        })

        # 그래프 그리기 함수 호출
        evaluation_metrics = show(allocation_dict, evaluation_target)
        return evaluation_metrics


    def visualize_performance(self):
        """ 성과 지표 시각화 """
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        
        # 누적 수익률 시각화
        self.portfolio_values["Cumulative Return"] = self.portfolio_values["Portfolio Value"] / self.portfolio_values["Portfolio Value"].iloc[0]
        ax[0].plot(self.portfolio_values.index, self.portfolio_values["Cumulative Return"], label="Cumulative Return")
        ax[0].set_title("Cumulative Return")
        ax[0].set_ylabel("Return")
        ax[0].legend()
        
        # MDD 시각화
        running_max = self.portfolio_values["Portfolio Value"].cummax()
        drawdown = (self.portfolio_values["Portfolio Value"] - running_max) / running_max
        ax[1].plot(self.portfolio_values.index, drawdown, label="Drawdown", color='red')
        ax[1].set_title("Maximum Drawdown (MDD)")
        ax[1].set_ylabel("Drawdown")
        ax[1].legend()

        plt.tight_layout()
        plt.show()