## import
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm


## Data

# 파일 호출
data_path: str = "C:/naverboostcamp/project1/data"
train_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "train.csv")).assign(_type="train") # train 에는 _type = train 
test_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")).assign(_type="test") # test 에는 _type = test
market_df: pd.DataFrame = pd.concat([train_df, test_df], axis=0)
network_df: pd.DataFrame = pd.concat([train_df, test_df], axis=0)
submission_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")) # ID, target 열만 가진 데이터 미리 호출

# HOURLY_ 로 시작하는 .csv 파일 이름을 file_names 에 할당
file_names_market: List[str] = [
    f for f in os.listdir(data_path) if f.startswith("HOURLY_MARKET-DATA") and f.endswith(".csv")
]

# 파일명 : 데이터프레임으로 딕셔너리 형태로 저장
file_dict_market: Dict[str, pd.DataFrame] = {
    f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f)) for f in file_names_market
}

# HOURLY_ 로 시작하는 .csv 파일 이름을 file_names 에 할당당
file_names_network: List[str] = [
    f for f in os.listdir(data_path) if f.startswith("HOURLY_NETWORK-DATA") and f.endswith(".csv")
]

# 파일명 : 데이터프레임으로 딕셔너리 형태로 저장
file_dict_network: Dict[str, pd.DataFrame] = {
    f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f)) for f in file_names_network
}
for _file_name, _df in tqdm(file_dict_market.items()): # (key, value)
    # 열 이름 중복 방지를 위해 {_file_name.lower()}_{col.lower()}로 변경, datetime 열을 ID로 변경
    _rename_rule = {
        col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID"
        for col in _df.columns
    }
    _df = _df.rename(_rename_rule, axis=1)
    market_df = market_df.merge(_df, on="ID", how="left")

for _file_name, _df in tqdm(file_dict_network.items()):  # (key, value)
    # 열 이름 중복 방지를 위해 {_file_name.lower()}_{col.lower()}로 변경, datetime 열을 ID로 변경
    _rename_rule = {
        col: f"{col.lower()}" if col != "datetime" else "ID"
            for col in _df.columns
        }
    _df = _df.rename(_rename_rule, axis=1)
    network_df = network_df.merge(_df, on="ID", how="left")


## EDA_market

# 결측치
# 각 열에서 누락된 값의 수를 계산
missing_values = market_df.isnull().sum()

# 누락된 값의 백분율 계산
missing_percentage = (missing_values / len(market_df)) * 100

# 누락된 값 비율을 기준으로 열 정렬
sorted_missing_percentage = missing_percentage.sort_values(ascending=False)
sorted_missing_percentage.value_counts().sort_index(ascending=False)

# 결측치 없는 칼럼만
drop_cols = list(sorted_missing_percentage[:53].index)
drop_cols.remove('target')

market_df = market_df.drop(columns=drop_cols)

# Mice
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge

def Mice(df):
    # target 제외
    X = df.drop(columns='target')
    y = df['target']

    # 숫자형 데이터만 선택
    numeric_df = X.select_dtypes(include=[np.number])
    exclude_numeric_df = X.select_dtypes(exclude=[np.number])

    # # 결측값을 평균값으로 사전 대체
    # simple_imputer = SimpleImputer(strategy='mean')
    # numeric_df_ = simple_imputer.fit_transform(numeric_df)

    # MICE 기법 적용
    mice_imputer = IterativeImputer(estimator=BayesianRidge())
    imputed_data = mice_imputer.fit_transform(numeric_df)

    # Numpy 배열을 DataFrame으로 변환
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns)

    # 원본 데이터의 string 타입 열을 그대로 합침
    final_df = pd.concat([imputed_df.reset_index(drop=True), exclude_numeric_df.reset_index(drop=True)], axis=1)

    # target 합침
    final_df = pd.concat([final_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    return final_df

market_df_final = Mice(market_df)


## EDA_network

# 결측치
# Mice (test)
from fancyimpute import IterativeImputer

def Mice(df):
    # target 제외
    X = df.drop(columns='target')
    y = df['target']

    # 숫자형 데이터만 선택
    numeric_df = X.select_dtypes(include=[np.number])
    exclude_numeric_df = X.select_dtypes(exclude=[np.number])

    # MICE 기법 적용
    mice_imputer = IterativeImputer()
    imputed_data = mice_imputer.fit_transform(numeric_df)

    # Numpy 배열을 DataFrame으로 변환
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns)

    # 원본 데이터의 string 타입 열을 그대로 합침
    final_df = pd.concat([imputed_df.reset_index(drop=True), exclude_numeric_df.reset_index(drop=True)], axis=1)

    # target 합침
    final_df = pd.concat([final_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    return final_df

network_df_final = Mice(network_df)


## Market, Network Concat
df = pd.concat([market_df_final.reset_index(drop=True), network_df_final.reset_index(drop=True).drop(columns=['target','ID','_type'])], axis=1)


## Feature Engineering
def feature_engineering(df):
    new_df = df[['ID','target','_type']]

    # 1. 청산
    new_df['long_liquidations_all'] = df['hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations']
    new_df['short_liquidations_all'] = df['hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations']

    new_df['long_liquidations_all_usd'] = df['hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations_usd']
    new_df['short_liquidations_all_usd'] = df['hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations_usd']

    new_df['long_liquidations_binance'] = df['hourly_market-data_liquidations_binance_all_symbol_long_liquidations']
    new_df['short_liquidations_binance'] = df['hourly_market-data_liquidations_binance_all_symbol_short_liquidations']

    new_df['long_liquidations_all_diff']=new_df['long_liquidations_all'] - new_df["short_liquidations_all"]
    new_df['long_liquidations_all_usd_diff']=new_df['long_liquidations_all_usd'] - new_df['short_liquidations_all_usd']

    new_df['long_liquidations_all_diffg']=np.sign(new_df['long_liquidations_all_diff'])
    new_df['long_liquidations_all_usd_diffg']=np.sign(new_df['long_liquidations_all_usd_diff'])


    ### rolling & shift
    new_df['long_liquidations_all_1d_avg'] = new_df['long_liquidations_all'].rolling(window=24).mean()
    new_df['long_liquidations_all_shift'] = new_df['long_liquidations_all'].shift(1)

    new_df['short_liquidations_all_1d_avg'] = new_df['short_liquidations_all'].rolling(window=24).mean()
    new_df['short_liquidations_all_shift'] = new_df['short_liquidations_all'].shift(1)


    # 2. 롱/숏 스퀴즈
    long_squeeze_threshold = new_df['short_liquidations_all'].mean() + 2 * new_df['short_liquidations_all'].std()
    short_squeeze_threshold = new_df['long_liquidations_all'].mean() + 2 * new_df['long_liquidations_all'].std()

    new_df['long_squeeze_all'] = (new_df['short_liquidations_all'] > long_squeeze_threshold).astype(int)  # 1 if short liquidations exceed threshold (long squeeze)
    new_df['short_squeeze_all'] = (new_df['long_liquidations_all'] > short_squeeze_threshold).astype(int)  # 1 if long liquidations exceed threshold (short squeeze)


    # 3. 미결제 약정
    new_df['open_interest_all'] = df['hourly_market-data_open-interest_all_exchange_all_symbol_open_interest']
    new_df['open_interest_binance'] = df['hourly_market-data_open-interest_binance_all_symbol_open_interest']

    ### 변화율
    new_df['open_interest_all_pct_change'] = new_df['open_interest_all'].pct_change() * 100  # Percentage change in Huobi open interest
    new_df['open_interest_binance_pct_change'] = new_df['open_interest_binance'].pct_change() * 100  # Percentage change in Binance open interest

    ### rolling & shift
    new_df['open_interest_all_1d_avg'] = new_df['open_interest_all'].rolling(window=24).mean()
    new_df['open_interest_all_shift'] = new_df['open_interest_all'].shift(1)


    # 4. 펀딩
    new_df['funding_reate_all'] = df['hourly_market-data_funding-rates_all_exchange_funding_rates']
    new_df['funding_rate_binance'] = df['hourly_market-data_funding-rates_binance_funding_rates']

    ### rolling
    new_df['funding_rate_1d_avg'] = new_df['funding_reate_all'].rolling(window=24).mean()
    new_df['funding_rate_7d_avg'] = new_df['funding_reate_all'].rolling(window=168).mean()

    ### 7 vs 30
    new_df['funding_rate_trend'] = (new_df['funding_rate_1d_avg'] > new_df['funding_rate_7d_avg']).astype(int)


    # 5. 매수/매도 ratio
    new_df['buy_all_ratio'] = df['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_ratio']
    new_df['sell_all_ratio'] = df['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_ratio']
    new_df['buy_sell_all_ratio'] = df['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_sell_ratio']


    ### 1이면 매수세가 우세(비율이 1 이하), 0이면 매도세가 우세(비율이 1 이상)
    new_df['buy_sell_all_ratio_dominance'] = (new_df['buy_sell_all_ratio'] > 1).astype(int)


    ### 변화율
    new_df['buy_sell_all_ratio_change'] = new_df['buy_sell_all_ratio'].pct_change() * 100


    ### rolling
    new_df['buy_sell_all_ratio_1d_avg'] = new_df['buy_sell_all_ratio'].rolling(window=24).mean()
    new_df['buy_sell_all_ratio_7d_avg'] = new_df['buy_sell_all_ratio'].rolling(window=168).mean()


    ### 7 vs 30
    new_df['buy_sell_all_ratio_trend'] = (new_df['buy_sell_all_ratio_1d_avg'] > new_df['buy_sell_all_ratio_7d_avg']).astype(int)


    # 6. 매수/매도 volume
    new_df['buy_all_volume'] = df['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume']
    new_df['sell_all_volume'] = df['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume']

    ### diff
    new_df['buy_sell_all_volume_diff'] = new_df["buy_all_volume"] - new_df["sell_all_volume"]
    new_df['buy_sell_all_volume_diffg']=np.sign(new_df["buy_all_volume"] - new_df["sell_all_volume"])


    # 7. 코인베이스
    new_df['coinbase_premium_gap'] = df['hourly_market-data_coinbase-premium-index_coinbase_premium_gap']
    new_df['coinbase_premium_index'] = df['hourly_market-data_coinbase-premium-index_coinbase_premium_index']


    # 8. 네트워크 데이터
    new_df[['hashrate','blockreward','difficulty','fees_reward_percent','fees_transaction_mean','supply_total','supply_new','transactions_count_total','tokens_transferred_total','velocity_supply_total']] = df[['hashrate','blockreward','difficulty','fees_reward_percent','fees_transaction_mean','supply_total','supply_new','transactions_count_total','tokens_transferred_total','velocity_supply_total']]

    # 9. 월, 일, 요일, 시간 변수 생성
    new_df['ID'] = pd.to_datetime(new_df['ID'])

    new_df['month'] = new_df['ID'].dt.month
    new_df['day'] = new_df['ID'].dt.day
    new_df['hour'] = new_df['ID'].dt.hour
    new_df['weekday']= new_df['ID'].dt.weekday
    new_df['weekend'] = 0
    new_df['weekend'][new_df['weekday']>=5] = 1
    
    return new_df

df = feature_engineering(df)


## Train, Test Split
train_df = df[df['_type']=='train']
test_df = df[df['_type']=='test']


## Model
X = train_df.drop(columns=['target','ID','_type'])
y = train_df['target'].astype('int')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def split_score_cm(model, X, y, n_splits=5):

    # 행과 열에 해당하는 클래스 이름 (0, 1, 2, 3)
    column_labels = ['Actual_0', 'Actual_1', 'Actual_2', 'Actual_3']
    index_labels = ['Pred_0', 'Pred_1', 'Pred_2', 'Pred_3']

    # 데이터프레임 생성
    data = np.zeros((4, 4))
    df = pd.DataFrame(data, index=index_labels, columns=column_labels)

    skf = StratifiedKFold(n_splits=n_splits)

    accuracy = []
    for train_index, val_index in skf.split(X, y):
        # 훈련 데이터와 테스트 데이터 나누기
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 모델 학습
        model.fit(X_train, y_train)

        # 모델 예측
        y_pred = model.predict(X_val)
        accuracy.append(accuracy_score(y_val, y_pred))

        # 4x4 혼동 행렬처럼 행은 실제값, 열은 예측값을 나타냄
        for i in range(len(y_pred)):
            df.iloc[y_pred[i], y_val.values[i]] += 1
    return np.mean(accuracy), df


## model
import xgboost as xgb

xgb_model = xgb.XGBClassifier(radom_state=42, n_estimators=500, max_depth=15)
acc_mean, cm =split_score_cm(xgb_model, X, y, n_splits=5)


## Feature Importance
import matplotlib.pyplot as plt

def feature_importance_split(model, X, k):
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})

    # Feature Importance 시각화
    xgb.plot_importance(model, max_num_features=k)
    plt.show()

    importance_col = list(importance_df['Feature'][importance_df['Importance'].sort_values(ascending=False)[:k].index])
    importance_col.append('_type')

    importance_df_train = df[importance_col][df['_type']=='train']
    importance_df_test = df[importance_col][df['_type']=='test']

    return importance_df_train, importance_df_test

importance_df_train, importance_df_test = feature_importance_split(xgb_model, X, 10)


## Predict
y_pred = xgb_model.predict_proba(test_df.drop(columns=['target','ID','_type']))
pd.DataFrame(y_pred).to_csv('JE.csv', index=False)
np.unique(y_pred, return_counts=True)


## output file 할당후 save 
submission_df = submission_df.assign(target = y_pred)
submission_df.to_csv("JE_pred.csv", index=False)