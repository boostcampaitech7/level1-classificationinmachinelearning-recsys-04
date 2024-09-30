import pandas as pd
import yaml
import sys
import os

# ID, target 열만 가진 데이터 미리 호출
data_path = "../data"
submission_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")) 

# 명령줄에서 첫 번째 인수로 받은 파일명을 사용하여 config 파일 열기
if len(sys.argv) < 2:
    print("Usage: python ensemble.py <config_file>")
    sys.exit(1)

config_file = sys.argv[1]

# config 파일을 읽어서 config 변수에 저장
with open(f'../config/{config_file}', 'r') as file:
    config = yaml.safe_load(file)

# CSV 파일 경로와 가중치 가져오기
csv_files = [(item['path'], item['weight']) for item in config['csv_files']]

# 가중치를 곱한 DataFrame 저장 변수 초기화
weighted_sum_df = None

# 각 csv 파일을 읽고 가중치 적용하여 더하기
for csv_path, weight in csv_files:

    # csv 파일 읽기
    df = pd.read_csv(csv_path)

    # 가중치 적용
    weighted_df = df * weight

    # DataFrame을 가중치가 적용된 결과에 더하기
    if weighted_sum_df is None:
        weighted_sum_df = weighted_df
    else:
        weighted_sum_df += weighted_df

# 각 행에서 가장 큰 값의 인덱스를 선택하여 예측 결과 생성
y_test_pred = weighted_sum_df.idxmax(axis=1)

# output file 할당후 save 
submission_df = submission_df.assign(target = y_test_pred)

# results 폴더에 yaml 파일명을 반영하여 저장
# yaml 파일명에서 확장자를 제거하고 csv 파일명 생성
output_filename = os.path.splitext(os.path.basename(config_file))[0] + ".csv"
output_path = os.path.join("../results", output_filename)

# 결과를 CSV 파일로 저장
submission_df.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")