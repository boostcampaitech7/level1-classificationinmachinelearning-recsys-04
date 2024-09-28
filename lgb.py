
### Library Import


import time
import pandas as pd
import os
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from lightgbm import LGBMClassifier


### Data Load

location = 'C:\\Users\\gagoo\\Desktop\\jupyter\\boost\\비트코인'

data_path: str = location+'\\data'
train_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "train.csv")).assign(_type="train") 
test_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")).assign(_type="test") 
submission_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")) 
df: pd.DataFrame = pd.concat([train_df, test_df], axis=0)

print('Data Load')

file_names: List[str] = [
    f for f in os.listdir(data_path) if f.startswith("HOURLY_") and f.endswith(".csv")
]


file_dict: Dict[str, pd.DataFrame] = {
    f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f)) for f in file_names
}

for _file_name, _df in tqdm(file_dict.items()):
    _rename_rule = {
        col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID"
        for col in _df.columns
    }
    _df = _df.rename(_rename_rule, axis=1)
    df = df.merge(_df, on="ID", how="left")


# btc  liquidation, open 에 대한 변수 추가
columns = df.columns
btc_columns = [col for col in columns if 'btc' in col]
btc_liq_columns = btc_columns[:76]
btc_open_columns = btc_columns[76:-2]
btc_open_df = df[btc_open_columns]
btc_liq_df = df[btc_liq_columns]
long = 0
short = 0 
for i in range(0,76,2):
    long += btc_liq_df.iloc[:,i].fillna(0)
    short += btc_liq_df.iloc[:,i+1].fillna(0)


# 모델에 사용할 컬럼, 컬럼의 rename rule을 미리 할당함
cols_dict: Dict[str, str] = {
    "ID": "ID",
    "target": "target",
    "_type": "_type",
    "hourly_market-data_funding-rates_all_exchange_funding_rates": "funding_rates",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_ratio": "buy_ratio",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_sell_ratio": "buy_sell_ratio",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume": "buy_volume",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_ratio": "sell_ratio",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume": "sell_volume",
    "hourly_network-data_addresses-count_addresses_count_active": "active_count",
    "hourly_network-data_addresses-count_addresses_count_receiver": "receiver_count",
    "hourly_network-data_addresses-count_addresses_count_sender": "sender_count",
    "hourly_network-data_tokens-transferred_tokens_transferred_total" : "transferred_total",
    "hourly_network-data_tokens-transferred_tokens_transferred_mean" : "transferred_mean",
    "hourly_network-data_fees_fees_total" : "fees_total",
}
df = df[cols_dict.keys()].rename(cols_dict, axis=1)


df['long_liq'] = long
df['short_liq'] = short
df['open'] = btc_open_df.iloc[:,:].sum(axis=1)


# 전처리 IQR 으로 전처리, nan값은 이동평균(3) 으로 채움


def replace_outliers_iqr(df, columns):
    for column in columns:
        # 1. IQR 계산
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # 2. 이상치 경계 설정
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 3. 이상치 대체
        df[column] = df[column].apply(
            lambda x: lower_bound if x < lower_bound else 
                        upper_bound if x > upper_bound else x
        )

    return df

def fill_nan_with_previous_mean(series):
    series = series.copy() 
    for i in range(len(series)):
        if pd.isna(series[i]) and i >= 2:
            series[i] = series[i-1:i-3:-1].mean()  
    return series
    

exclude_cols = ['_type','target', 'ID']
outlier_col = [col for col in df.columns if col not in exclude_cols]

df = replace_outliers_iqr(df, outlier_col)

df_nan_col = df.columns[df.isna().any()].tolist()
df_nan_col.remove('target')
df = df.reset_index().drop('index',axis=1)
for col in df_nan_col :
    df[col] = fill_nan_with_previous_mean(df[col])
    

# 시간 변수 추가

df['ID'] = pd.to_datetime(df['ID'])
df['hour_sin'] = np.sin(2 * np.pi * df['ID'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['ID'].dt.hour / 24)
df['weekday'] = df['ID'].dt.weekday.apply(lambda x: 0 if x >= 5 else 1)


# 일반적인 변수 추가 
df['total_volume']= df['buy_volume']+df['sell_volume']
df['change_volume'] = df['buy_volume'] - df['sell_volume']
df['total_volume_sign']= np.sign(df['buy_volume']-df['sell_volume'])
df['total_liq'] = df['long_liq'] + df['short_liq']
df['change_liq']=df["long_liq"] - df["short_liq"]
df['liq_sign']=np.sign(df["long_liq"] - df["short_liq"])
df['long_liq_ratio'] = df['long_liq'] / df['total_liq']
df['short_liq_ratio'] = df['short_liq'] / df['total_liq']

df['open_diff'] = df['open'].diff()


# 시간에 따른 변수 추가 함수
def make_volume_col(dataframe,make_volume_list,window_size):
    for i in make_volume_list :
        dataframe[str(i)+'_diff'] = dataframe[i].diff()
        dataframe[str(i)+'_diff'+str(window_size)] = dataframe[i].diff(window_size)
        dataframe[str(i)+'_sma'+str(window_size)] = dataframe[i].rolling(window=window_size).mean()
        dataframe[str(i)+'_std'+str(window_size)] = dataframe[i].rolling(window=window_size).std()
    return dataframe        

volume_col = ['buy_volume','sell_volume','total_volume','change_volume',
              'long_liq','short_liq','total_liq','change_liq']

window_size = 4

df = make_volume_col(df,volume_col,window_size)


# 시간에 따른 변화율 추가 함수
def make_ratio_col(dataframe,make_ratio_list,widosw_size):
    for i in make_ratio_list :
        if i in 'liq' :
            dataframe[str(i)+'_ratio_diff'] = dataframe[str(i)+'_diff']  / ( dataframe['total_liq_diff']+1)
            dataframe[str(i)+'_ratio_diff'+str(window_size)] = dataframe[str(i)+'_diff'+str(window_size)] / ( dataframe['total_liq_diff'+str(window_size)] +1)
            dataframe[str(i)+'_ratio_sma'+str(window_size)] = dataframe[str(i)+'_sma'+str(window_size)] / ( dataframe['total_liq_sma'+str(window_size)] +1)
        else :
            dataframe[str(i)+'_ratio_diff'] = dataframe[str(i)+'_diff']  / ( dataframe['total_volume_diff'] + 1)
            dataframe[str(i)+'_ratio_diff'+str(window_size)] = dataframe[str(i)+'_diff'+str(window_size)] / ( dataframe['total_volume_diff'+str(window_size)] + 1 )
            dataframe[str(i)+'_ratio_sma'+str(window_size)] = dataframe[str(i)+'_sma'+str(window_size)] / ( dataframe['total_volume_sma'+str(window_size)] + 1 ) 
            
    return dataframe 



ratio_col = ['buy_volume','sell_volume','change_volume',
              'long_liq','short_liq','change_liq']

df = make_ratio_col(df,ratio_col,window_size)

features = list(df.columns)
to_remove  = ['ID']
features = [feature for feature in features if feature not in to_remove]
df = df[features]
df = df.iloc[24:,:]
df = df.fillna(0)

### 모델링

train_df = df.loc[df["_type"]=="train"].drop(columns=["_type"])
test_df = df.loc[df["_type"]=="test"].drop(columns=["_type"])

X_train, X_valid, y_train, y_valid = train_test_split(
    train_df.drop(["target"], axis=1), 
    train_df["target"].astype(int), 
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_train)
X_train = pd.DataFrame(scaled_data,columns=X_train.columns)
X_valid = pd.DataFrame(scaler.transform(X_valid),columns=X_valid.columns)

param_grid = {
    'n_estimators': [5,10,30,50,100],
    'min_child_samples': [2,3,5],  
    'num_leaves': [10,20,30,40],    
    'learning_rate': [0.01,0.05, 0.1],  
    'boosting_type': ['gbdt', 'dart'], 
    'colsample_bytree': [0.6, 0.8, 1.0],  
}

param_list = list(ParameterGrid(param_grid))
total_fits = len(param_list) * 5 


# 반복문 변수 설정
best_score = -np.inf 
best_params = None  
best_model = None  

print('Grid Search Start')
ct = time.time()
with tqdm(total=total_fits) as pbar:
    
    for params in param_list:
        fold_scores = []  
        
        for fold in range(5): 
            
            lgbm = LGBMClassifier(**params, random_state=42, device='gpu',verbose=0,num_class=4)
            lgbm.fit(X_train, y_train)
            
            
            y_pred = lgbm.predict(X_valid)  
            score = accuracy_score(y_valid, y_pred)  
            fold_scores.append(score)

            pbar.update(1)  

        mean_score = np.mean(fold_scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_model = lgbm


print('\nmodeling time :\n', time.time()-ct)
print(f"Best Score: {best_score}")
print(f"Best Parameters: {best_params}")


X_valid = pd.DataFrame(X_valid,columns=X_valid.columns)
y_pred = best_model.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

scaled_test = pd.DataFrame(scaler.transform(test_df[X_train.columns]),columns = X_train.columns)
predict = best_model.predict_proba(scaled_test)
pd.DataFrame(predict,columns=['0','1','2','3',]).to_csv('sm_pred.csv',index=False)

print('Finish')



