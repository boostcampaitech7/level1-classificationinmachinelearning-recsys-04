import os
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# 파일 호출
data_path: str = "C:/Users/ssk07/OneDrive/바탕 화면/코드백업/data/data"
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

for _file_name, _df in tqdm(file_dict.items()):
    # 열 이름 중복 방지를 위해 {_file_name.lower()}_{col.lower()}로 변경, datetime 열을 ID로 변경
    _rename_rule = {
        col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID"
        for col in _df.columns
    }
    _df = _df.rename(_rename_rule, axis=1)
    df = df.merge(_df, on="ID", how="left")

# Feature engineering
df['ID'] = pd.to_datetime(df['ID'])

# 연, 월, 일로 분리
df['Year'] = df['ID'].dt.year
df['Month'] = df['ID'].dt.month
df['Day'] = df['ID'].dt.day

# 요일 추가 (월요일=0, 일요일=6)
df['Weekday'] = df['ID'].dt.dayofweek

# 혹은 요일을 문자열로 표시
df['Weekday_str'] = df['ID'].dt.strftime('%A')

# 모델에 사용할 컬럼, 컬럼의 rename rule을 미리 할당함
cols_dict: Dict[str, str] = {
    "ID": "ID",
    "target": "target",
    "_type": "_type",
    "hourly_market-data_coinbase-premium-index_coinbase_premium_gap": "coinbase_premium_gap",
    "hourly_market-data_coinbase-premium-index_coinbase_premium_index": "coinbase_premium_index",
    "hourly_market-data_funding-rates_all_exchange_funding_rates": "funding_rates",
    "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations": "long_liquidations",
    "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations_usd": "long_liquidations_usd",
    "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations": "short_liquidations",
    "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations_usd": "short_liquidations_usd",
    "hourly_market-data_open-interest_all_exchange_all_symbol_open_interest": "open_interest",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_ratio": "buy_ratio",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_sell_ratio": "buy_sell_ratio",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume": "buy_volume",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_ratio": "sell_ratio",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume": "sell_volume",
    "hourly_network-data_addresses-count_addresses_count_active": "active_count",
    "hourly_network-data_addresses-count_addresses_count_receiver": "receiver_count",
    "hourly_network-data_addresses-count_addresses_count_sender": "sender_count",
}
df = df[cols_dict.keys()].rename(cols_dict, axis=1)

# 이상치 처리

def replace_outliers_with_previous(df, column_list):
    # column_list는 이상치를 대체할 열의 목록
    for column in column_list:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 이상치를 감지하고 이전 행의 값으로 대체
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        df.loc[outliers, column] = df.loc[outliers, column].shift(1)

        # 첫 행이 이상치인 경우 중앙값으로 대체
        if pd.isna(df.loc[0, column]) and outliers.iloc[0]:
            df.loc[0, column] = df[column].median()  # 중앙값으로 대체

    return df

# 숫자형 열만 선택
numeric_cols = df.select_dtypes(include=['number']).columns

# 이상치 대체 실행
df_clean = replace_outliers_with_previous(df, numeric_cols)
df = df_clean

# MICE

# 1. ID와 같은 비수치형 열은 제외하고 수치형 열만 선택
numeric_df = df.select_dtypes(include=[float, int])

# 2. IterativeImputer 초기화 (DecisionTreeRegressor 사용하여 MICE 수행)
imputer = IterativeImputer(estimator=DecisionTreeRegressor(), max_iter=10, random_state=0)

# 3. 수치형 열에 대해서만 결측값 대체 수행
imputed_numeric_df = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns, index=numeric_df.index)

# 4. 결측값 대체 후 원래 데이터프레임과 결합
df_cleaned = df.copy()  # 원래 데이터프레임 복사
df_cleaned[numeric_df.columns] = imputed_numeric_df  # 대체된 수치형 열을 원래 데이터프레임에 적용

df = df_cleaned

# eda 에서 파악한 차이와 차이의 음수, 양수 여부를 새로운 피쳐로 생성
df = df.assign(
    liquidation_diff=df["long_liquidations"] - df["short_liquidations"],
    liquidation_usd_diff=df["long_liquidations_usd"] - df["short_liquidations_usd"],
    volume_diff=df["buy_volume"] - df["sell_volume"],
    liquidation_diffg=np.sign(df["long_liquidations"] - df["short_liquidations"]),
    liquidation_usd_diffg=np.sign(df["long_liquidations_usd"] - df["short_liquidations_usd"]),
    volume_diffg=np.sign(df["buy_volume"] - df["sell_volume"]),
    buy_sell_volume_ratio=df["buy_volume"] / (df["sell_volume"] + 1),
)
# category, continuous 열을 따로 할당해둠
category_cols: List[str] = ["liquidation_diffg", "liquidation_usd_diffg", "volume_diffg"]
conti_cols: List[str] = [_ for _ in cols_dict.values() if _ not in ["ID", "target", "_type"]] + [
    "buy_sell_volume_ratio",
    "liquidation_diff",
    "liquidation_usd_diff",
    "volume_diff",
]

def shift_feature(
    df: pd.DataFrame,
    conti_cols: List[str],
    intervals: List[int],
) -> List[pd.Series]:
    """
    연속형 변수의 shift feature 생성
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
        intervals (List[int]): shifted intervals
    Return:
        List[pd.Series]
    """
    df_shift_dict = [
        df[conti_col].shift(interval).rename(f"{conti_col}_{interval}")
        for conti_col in conti_cols
        for interval in intervals
    ]
    return df_shift_dict

# 최대 24시간의 shift 피쳐를 계산
shift_list = shift_feature(
    df=df, conti_cols=conti_cols, intervals=[_ for _ in range(1, 24)]
)

df.loc[df['_type'] == 'test', 'target'] = None

# concat 하여 df 에 할당
df = pd.concat([df, pd.concat(shift_list, axis=1)], axis=1)

df_target = df['target']

# scaling
# 범주형 컬럼을 제외한 수치형 컬럼만 선택
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns

# Min-Max Scaling 적용
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 범주형 데이터와 스케일링된 수치형 데이터를 합침
df_scaled = pd.concat([df[numeric_columns], df[categorical_columns]], axis=1)

df_scaled['target'] = df_target
# 결과 확인
df_scaled.head()

df = df_scaled

# _type에 따라 train, test 분리
train_df = df.loc[df["_type"]=="train"].drop(columns=["_type"])
test_df = df.loc[df["_type"]=="test"].drop(columns=["_type"])

# Model Training

# 데이터 준비
X = train_df.drop(["target", "ID"], axis=1)
y = train_df["target"].astype(int)

# TimeSeriesSplit으로 데이터 나누기 (5개의 폴드로 나눔)
tscv = TimeSeriesSplit(n_splits=5)

# XGBoost 파라미터 설정
params = {
    "objective": "multi:softprob",  # 다중 클래스 문제
    "eval_metric": "mlogloss",      # 다중 클래스 로그 손실
    "num_class": 4,                 # 클래스 수
    "learning_rate": 0.05,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

# 폴드별로 학습 및 검증
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}")
    
    # 학습 및 검증 데이터 분리
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    # XGBoost DMatrix 생성
    train_data = xgb.DMatrix(X_train, label=y_train)
    valid_data = xgb.DMatrix(X_valid, label=y_valid)
    
    # XGBoost 학습
    evals = [(train_data, "train"), (valid_data, "valid")]
    xgb_model = xgb.train(
        params=params, 
        dtrain=train_data, 
        num_boost_round=50, 
        evals=evals, 
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # 검증 데이터로 예측
    y_valid_pred = xgb_model.predict(valid_data)
    y_valid_pred_class = np.argmax(y_valid_pred, axis=1)
    
    # 성능 평가
    accuracy = accuracy_score(y_valid, y_valid_pred_class)
    
    print(f"Fold {fold+1} - Accuracy: {accuracy}")

# performance 체크 후 전체 학습 데이터로 다시 재학습
x_train = xgb.DMatrix(train_df.drop(["target", "ID"], axis=1), label=train_df["target"].astype(int))
xgb_model = xgb.train(
    params=params, 
    dtrain=x_train, 
    num_boost_round=xgb_model.best_iteration
)

# Inference, Output save

# DataFrame을 DMatrix로 변환
dtest = xgb.DMatrix(test_df.drop(["target", "ID"], axis=1))

# 각 클래스에 대한 확률값 예측
y_test_pred_proba = xgb_model.predict(dtest)

# 다중 클래스 예측을 위한 argmax 사용 (가장 높은 확률을 가진 클래스 선택)
y_test_pred_class = np.argmax(y_test_pred_proba, axis=1)

# 확률값을 데이터프레임에 추가
prob_df = pd.DataFrame(y_test_pred_proba, columns=[f'{i}' for i in range(y_test_pred_proba.shape[1])])

# 결과 저장
prob_df.to_csv("jw_pred2.csv", index=False)
