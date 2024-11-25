from Tree import *
from BaseOptimizer import *
from Optimizer import *
from Assumption import *
from Pipeline import *
from Backtest import *
from Evaluation import *
from main import *
import time
import warnings

warnings.filterwarnings('ignore')

def rich_func(invester_rank: int, assets: list, investor_goal: int = 4):
    start = time.time()

    # 호윤부분

    # 호윤 Tree 반환 완료

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
    prev_date = (datetime.today() - timedelta(days=252 * 10)).strftime("%Y-%m-%d")
    rebalance_dates = pd.date_range(prev_date, today, freq='M')
    trading_days = price_data.index

    backtest = Backtest(pipe, price_data, rebalance_dates, trading_days)
    backtest.run_backtest()

    allocation = backtest.allocations[-1][-1]

    # TODO json 뱉을 때, 누적수익, MDD 시계열 데이터로 JSON에 같이 
    eval_metrix = backtest.evaluation()
