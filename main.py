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

def build_investment_tree(codes: list, risk_level: int, df: pd.DataFrame) -> Tree:
    tree = Tree("Universe")
    
    # 위험등급별 weight_bounds 설정
    if risk_level in [1,2]:  # 안정추구형
        bounds = {
            5: (0, 0.15),
            4: (0, 0.4),
            3: (0, 0.2),
            2: (0, 1),
            1: (0, 1)
        }
    elif risk_level == 3:  # 위험중립형
        bounds = {
            5: (0, 0.2),
            4: (0, 0.8),
            3: (0, 1),
            2: (0, 1),
            1: (0, 1)
        }
    else:  # 적극투자형 (4, 5)
        bounds = {
            5: (0, 0.2),
            4: (0, 1),
            3: (0, 1),
            2: (0, 1),
            1: (0, 1)
        }

    # 첫번째 열과 종목코드 매핑 생성
    row_code_dict = {}
    for _, row in df.iterrows():
        row_code_dict[row.iloc[0]] = str(row['종목 코드'])
    
    #print(row_code_dict)
    # 먼저 각 코드의 level 정보 수집
    code_info = {}
    for code in codes:
        print(code)
        row = df[df['종목 코드'].astype(str) == str(code)].iloc[0]
        code_info[code] = {
            'level': row['level'] if not pd.isna(row['level']) else 0,
            'parent': row['부모'] if not pd.isna(row['부모']) else None,
            'risk': row['투자 가능 대상 여부']
        }
    
    sorted_codes = sorted(codes, key=lambda x: code_info[x]['level'], reverse=True)
    for code in sorted_codes:
        bound = bounds[code_info[code]['risk']]
        if code_info[code]['level'] != 2:
            if row_code_dict[code_info[code]['parent']] != code:
                if row_code_dict[code_info[code]['parent']] not in [node.name for node in tree.get_all_nodes()]:
                    tree.insert('Universe', row_code_dict[code_info[code]['parent']], weight_bounds=bound)
                    tree.insert(row_code_dict[code_info[code]['parent']], code, weight_bounds=bound)
                    
                elif row_code_dict[code_info[code]['parent']] in [node.name for node in tree.get_all_nodes()]:
                    tree.insert(row_code_dict[code_info[code]['parent']], code, weight_bounds=bound)
                
            else:
                if code not in [node.name for node in tree.get_all_nodes()]:
                    tree.insert('Universe', code, weight_bounds=bound)
                    tree.insert(code, code, weight_bounds=bound)
                elif code not in [node.name for node in tree.get_all_nodes()]:
                    tree.insert(code, code, weight_bounds=bound)
                    
        elif code_info[code]['level'] == 2:
            tree.insert(code, code, weight_bounds=bound)
            
    
    tree.draw()
    return tree

def main(risk_level: int = 4, investor_goal:int = 1):

    file_path = 'invest_universe.csv'
    # 종목코드-종목설명 딕셔너리 생성

    universe = pd.read_csv(file_path, encoding='cp949')

    stock_dict = {}
    for _, row in universe.iterrows():
        code = str(row['종목 코드'])
        if len(code) < 6:
            code = '0' * (6 - len(code)) + code
        stock_dict[row['종목 설명']] = code

    universe['종목 코드'] = stock_dict.values()
    # price_data = pd.read_csv(file_path, index_col=0, parse_dates=True)

    codes = ['069500','139260','161510','273130','439870','251340','114260']

    #TODO 호윤 화이팅 해야하는 부분
    tree = build_investment_tree(codes, risk_level, universe)

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