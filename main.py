from Tree import *
from BaseOptimizer import *
from Optimizer import *
from Assumption import *
from Pipeline import *
from Backtest import *
from DataReader import *
from datetime import datetime, timedelta
import warnings 

warnings.filterwarnings('ignore')

def main(investor_goal:int = 1):

    # file_path = 'db.csv'
    # price_data = pd.read_csv(file_path, index_col=0, parse_dates=True)

    #TODO 호윤 화이팅 해야하는 부분
    tree = Tree("Universe")
    tree.insert("Universe", "069500", weight_bounds=(0.1, 0.9))
    tree.insert("Universe", "139260", weight_bound=(0.05, 0.8))
    tree.insert("069500", "161510", weight_bounds=(0.1, 0.9))
    tree.insert("069500", "091170", weight_bounds=(0.1, 0.9))
    tree.insert("139260", "325010", weight_bounds=(0.0, 0.9))
    tree.insert("139260", "252670", weight_bounds=(0.0, 0.9))
    tree.draw()

    assets = tree.get_all_nodes_name()
    price_data = fetch_data_from_db(assets)

    assumption = AssetAssumption(returns_window=52, covariance_window=52)

    ## 투자자 목표 1: 목돈 마련
    if investor_goal == 1:
        steps = [
            ("SAA", dynamic_risk_optimizer), 
            ("TAA", goal_based_optimizer),
            ("AP", mean_variance_optimizer)
        ]

    ## 투자자 목표 2: 결혼자금 준비
    elif investor_goal == 2:
        steps = [
            ("SAA", dynamic_risk_optimizer), 
            ("TAA", mean_variance_optimizer),
            ("AP", mean_variance_optimizer)
        ]

    ## 투자자 목표 3: 노후자금 준비
    elif investor_goal == 3:
        steps = [
            ("SAA", risk_parity_optimizer), 
            ("TAA", goal_based_optimizer),
            ("AP", mean_variance_optimizer)
        ]

    ## 투자자 목표 4: 장기 수익 창출
    else:
        steps = [
            ("SAA", risk_parity_optimizer), 
            ("TAA", mean_variance_optimizer),
            ("AP", mean_variance_optimizer)
        ]

    pipe = Pipeline(steps, tree, assumption)
    
    today = datetime.today().strftime("%Y-%m-%d")
    prev_date = (datetime.today() - timedelta(days=252 * 1)).strftime("%Y-%m-%d")
    rebalance_dates = pd.date_range(prev_date, today, freq='M')
    trading_days = price_data.index

    backtest = Backtest(pipe, price_data, rebalance_dates, trading_days)
    backtest.run_backtest()

    allocation = backtest.allocations[-1][-1]

    # TODO json 뱉을 때, 누적수익, MDD 시계열 데이터로 JSON에 같이 
    eval_metrix = backtest.evaluation(allocation)
    print(eval_metrix)

    # 성과 계산 및 시각화
    # cumulative_return, mdd, sharpe_ratio = backtest.calculate_performance()
    # print(f"Cumulative Return: {cumulative_return:.2%}")
    # print(f"Maximum Drawdown (MDD): {mdd:.2%}")
    # print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # 성과 시각화
    # backtest.visualize_performance()

main()