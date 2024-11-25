import FinanceDataReader as fdr
import pandas as pd
import sqlite3
from datetime import datetime

# 함수 정의: Tickers를 기준으로 데이터 가져오기
def fetch_close_prices():
    """
    여러 종목의 종가 데이터를 하나의 DataFrame으로 저장
    :param tickers: 종목 티커 리스트
    :param start_date: 데이터 시작 날짜
    :param end_date: 데이터 종료 날짜
    :return: 하나의 DataFrame (날짜 기준 정렬, 결측값 포함)
    """
    start_date='2000-01-01'
    # 오늘 날짜를 'yy-mm-dd' 형식으로 가져오기
    end_date = datetime.today().strftime('%y-%m-%d')

    # 파일 경로
    file_path = 'invest_universe.csv'

    # SQLite3 데이터베이스 경로
    db_path = 'financial_data.db'

    # 데이터 로드
    tickers_data = pd.read_csv(file_path, encoding='cp949')  # 한글 인코딩 처리

    # '종목 코드' 컬럼을 문자열로 변환하고 6자리로 맞추기
    tickers_data['종목 코드'] = tickers_data['종목 코드'].apply(lambda x: f"{int(x):06}")

    # Ticker 리스트 추출
    tickers = tickers_data['종목 코드']

    all_close_prices = pd.DataFrame()
    
    for ticker in tickers:
        try:
            # 티커 데이터 읽기
            df = fdr.DataReader(ticker, start=start_date, end=end_date)
            # 인덱스를 yyyy-mm-dd 형식으로 변환
            df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
            # 종가 데이터만 추출
            close_prices = df['Close']
            # 컬럼 이름 변경 (티커명)
            close_prices.name = ticker
            # 데이터프레임 병합 (outer join)
            all_close_prices = pd.concat([all_close_prices, close_prices], axis=1)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    return all_close_prices

def fetch_data_from_db(tickers, db_path='financial_data.db'):
    """
    SQLite3 데이터베이스에서 종목 리스트에 해당하는 시계열 데이터를 조회하여 DataFrame으로 반환
    :param tickers: 조회할 종목 리스트
    :param db_path: SQLite3 데이터베이스 경로
    :return: 선택한 종목들의 시계열 데이터를 포함한 DataFrame
    """
    try:
        # 데이터베이스 연결
        conn = sqlite3.connect(db_path)

        # 컬럼 이름을 대괄호로 감싸서 쿼리 작성
        columns = ', '.join([f'"{ticker}"' for ticker in tickers])
        query = f"""
        SELECT Date, {columns}
        FROM db
        """

        # 데이터베이스에서 데이터 읽기
        df = pd.read_sql_query(query, conn, index_col='Date')
        conn.close()

        # 인덱스를 datetime으로 변환
        df.index = pd.to_datetime(df.index)
        return df

    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return pd.DataFrame()