import os
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

# 파일 호출
data_path: str = "../../data"
train_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "train.csv")).assign(_type="train") # train 에는 _type = train 
test_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")).assign(_type="test") # test 에는 _type = test
submission_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")) # ID, target 열만 가진 데이터 미리 호출
df: pd.DataFrame = pd.concat([train_df, test_df], axis=0)

# HOURLY_ 로 시작하는 .csv 파일 이름을 file_names 에 할딩
file_names: List[str] = [
    f for f in os.listdir(data_path) if f.startswith("HOURLY_") and f.endswith(".csv")
]

# 파일명 : 데이터프레임으로 딕셔너리 형태로 저장
file_dict: Dict[str, pd.DataFrame] = {
    f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f)) for f in file_names
}

# 파일명 : 데이터프레임으로 딕셔너리 형태로 저장
for _file_name, _df in tqdm(file_dict.items()):
    # 열 이름 중복 방지를 위해 {_file_name.lower()}_{col.lower()}로 변경, datetime 열을 ID로 변경
    _rename_rule = {
        col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID"
        for col in _df.columns
    }
    _df = _df.rename(_rename_rule, axis=1)
    df = df.merge(_df, on="ID", how="left")

# 주요 10가지 특성을 바탕으로 시계열 특성(lag, rolling mean)을 추가하고 NaN 값 처리하는 코드
# 주요 10가지 특성 선택 
selected_features = [
    'hourly_market-data_liquidations_binance_all_symbol_long_liquidations_usd',
    'hourly_market-data_liquidations_binance_all_symbol_long_liquidations',
    'hourly_market-data_liquidations_binance_btc_usdt_long_liquidations',
    'hourly_market-data_liquidations_binance_btc_usdt_long_liquidations_usd',
    'hourly_market-data_liquidations_binance_btc_usd_long_liquidations',
    'hourly_market-data_liquidations_binance_btc_usd_long_liquidations_usd',
    'hourly_market-data_taker-buy-sell-stats_binance_taker_sell_ratio',
    'hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_ratio',
    'hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations',
    'hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations_usd'
]

# _type이 train인 데이터만 추출
train_data_filtered = df[df['_type'] == 'train']
train_data_filtered = train_data_filtered[['target'] + selected_features]

# 1시간에서 2시간까지의 lag를 추가
for feature in selected_features:
    for lag in range(1, 4):  # 1-hour to 3-hour lag
        train_data_filtered[f'{feature}_lag_{lag}'] = train_data_filtered[feature].shift(lag)

# 2시간의 이동 평균 추가
for feature in selected_features:
    train_data_filtered[f'{feature}_rolling_mean_2'] = train_data_filtered[feature].rolling(window=2).mean()
    train_data_filtered[f'{feature}_rolling_mean_3'] = train_data_filtered[feature].rolling(window=3).min()

# Forward fill을 사용하여 NaN 값을 처리
train_data_filtered.ffill()

# Forward fill을 진행한 후 남은 NaN 값을 -999로 처리
train_data_filtered.fillna(-999, inplace=True)

# 데이터를 train/test로 나누기
X = train_data_filtered.drop(columns=['target'])
y = train_data_filtered['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

model = CatBoostClassifier(loss_function='MultiClass',
                            eval_metric='Accuracy',
                            learning_rate=0.001,
                            iterations=100,
                            depth=6,
                            random_strength=0,
                            l2_leaf_reg=1.0,
                            task_type='GPU', # GPU를 사용하도록 설정
                            random_seed=42,
                            )
model.fit(X_train, y_train)

test_data_filtered = df[df['_type'] == 'test']
test_data_filtered = test_data_filtered[selected_features]

for feature in selected_features:
    for lag in range(1, 4):  # 1-hour to 3-hour lag
        test_data_filtered[f'{feature}_lag_{lag}'] = test_data_filtered[feature].shift(lag)
        
for feature in selected_features:
    test_data_filtered[f'{feature}_rolling_mean_2'] = test_data_filtered[feature].rolling(window=2).mean()
    test_data_filtered[f'{feature}_rolling_mean_3'] = test_data_filtered[feature].rolling(window=3).min()

# Train 데이터와 동일하게 NaN 값 처리
test_data_filtered.ffill()
test_data_filtered.fillna(-999, inplace=True)

test_predictions_proba_model = model.predict_proba(test_data_filtered)

hj_pred = pd.DataFrame(test_predictions_proba_model)

hj_pred.to_csv("../../results/hj_pred2.csv", index=False)