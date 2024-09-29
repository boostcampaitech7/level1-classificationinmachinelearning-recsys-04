# 🚀 Bitcoin Price Prediction Project

## 📌 프로젝트 개요
본 프로젝트는 비트코인의 다음 시점(한 시간 뒤)에서의 가격 등락을 예측하는 모델을 개발하는 것을 목표로 했습니다. 시계열 데이터 분석, 머신러닝 모델링, 금융 데이터 처리 등 다양한 기술을 실제 문제에 적용하여 비트코인의 가격 변동성을 예측하는 것이 주요 과제였습니다.

주어진 네트워크 데이터와 시장 데이터를 바탕으로 비트코인의 가격 등락을 다음 네 가지 범주로 예측합니다.
- 📉 -0.5% 미만
- 📉 -0.5% 이상 0% 미만
- 📈 0% 이상 0.5% 미만
- 📈 0.5% 이상

## 📊 프로젝트 데이터
본 프로젝트에서 사용된 데이터는 크립토퀀트에서 제공한 블록체인의 온체인 데이터입니다. 이 데이터는 네트워크 데이터와 시장 데이터로 구성되어 있으며, 총 107개의 특성 데이터 파일이 포함되어 있습니다. 각 파일은 시간 간격, 카테고리, 엔드포인트, 거래소/심볼 정보를 포함하고 있습니다.

- **학습 데이터:** 2023년 1월 1일부터 12월 31일까지의 시간별 데이터 (총 8,760개)
- **평가 데이터:** 2024년 1월 1일부터 4월 26일까지의 시간별 데이터 (총 2,792개)


## 🗂️ 파일 구조
```
.
├── README.md
├── config
│   ├── ensemble_submission_1.yaml
│   └── ensemble_submission_2.yaml
├── data
│   ├── lgb_under3_count_features.csv
├── results
│   ├── JE_pred.csv
│   ├── ensemble_submission_1.csv
│   ├── ensemble_submission_2.csv
│   ├── hj_pred1.csv
│   ├── hj_pred2.csv
│   ├── jw_pred1.csv
│   ├── jw_pred2.csv
│   ├── sm_pred.csv
│   └── sy_pred.csv
└── src
    ├── ensemble.py
    └── models
        ├── catboost-model.py
        ├── catboost-model2.py
        ├── final_JE.py
        ├── jw_pred1.py
        ├── jw_pred2.py
        ├── lgb-model.py
        └── lgb.py
```

### 폴더 및 파일 설명
- **config 폴더**  
  `ensemble_submission_1.yaml`, `ensemble_submission_2.yaml` 파일들은 `ensemble.py`를 실행할 때 사용하는 YAML 파일입니다. 앙상블하고 싶은 CSV 파일과 각 모델에 할당할 가중치가 적혀 있습니다.

- **data 폴더**  
  `lgb_under3_count_features.csv`는 데이터 전처리 과정에서 사용된 CSV 파일로, 특정 모델에 필요한 특징 값을 포함하고 있습니다.

- **results 폴더**  
  이 폴더에는 각 모델이 예측한 결과 파일들이 저장됩니다. 각 CSV 파일은 모델별로 다르게 생성되며, 이를 바탕으로 최종 앙상블 결과를 도출합니다.
  
  - `JE_pred.csv`, `ensemble_submission_1.csv`, `ensemble_submission_2.csv`, `hj_pred1.csv`, `hj_pred2.csv`, `jw_pred1.csv`, `jw_pred2.csv`, `sm_pred.csv`, `sy_pred.csv`: 각 모델의 예측 결과가 저장된 파일들입니다.

- **src 폴더**  
  이 폴더에는 프로젝트의 핵심 Python 코드가 포함되어 있습니다.
  
  - `ensemble.py`: 여러 모델의 예측 결과를 soft voting 방식으로 앙상블해주는 코드입니다. YAML 파일을 읽어와 가중치와 함께 예측을 진행합니다.
  
  - `models` 폴더: 각종 머신러닝 모델들이 구현된 파일들이 들어 있습니다.
  
    - **catboost-model.py**: CatBoost를 바탕으로 만든 모델입니다. 세부 내용은 [#2 PR](https://github.com/boostcampaitech7/level1-classificationinmachinelearning-recsys-04/pull/2)에서 확인할 수 있습니다.
    
    - **final_JE.py**: XGBoost를 바탕으로 만든 모델입니다. 자세한 내용은 [#3 PR](https://github.com/boostcampaitech7/level1-classificationinmachinelearning-recsys-04/pull/3)을 참고하세요.
    
    - **jw_pred1.py**: XGBoost 기반 모델로, 자세한 내용은 [#6 PR](https://github.com/boostcampaitech7/level1-classificationinmachinelearning-recsys-04/pull/6)에서 확인할 수 있습니다.
    
    - **lgb-model.py**: LightGBM을 사용한 모델로, 세부 사항은 [#5 PR](https://github.com/boostcampaitech7/level1-classificationinmachinelearning-recsys-04/pull/5)에서 확인 가능합니다.
    
    - **lgb.py**: LightGBM 기반의 또 다른 모델로, 추가 정보는 [#7 PR](https://github.com/boostcampaitech7/level1-classificationinmachinelearning-recsys-04/pull/7)에서 확인할 수 있습니다.


## 🛠️ 사용 방법
1. **개별 모델 실행:**  
   `src/models` 폴더에 존재하는 각 모델을 실행하면 예측 결과 파일(`pred.csv`)이 생성됩니다.

    ```
    python model_name.py
    ```
2. **앙상블(Ensemble) 실행:**  
    `src/ensemble.py`는 각 모델이 예측한 `csv` 파일을 읽어 soft voting 방식으로 앙상블을 진행합니다.  

    앙상블 실행 시, `config` 폴더에 있는 YAML 파일을 입력으로 제공해야 하며, 해당 YAML 파일은 예측에 사용될 CSV 파일과 각 파일의 가중치를 정의합니다.  

    ```
    python ensemble.py yaml_file_name.yaml
    ```
3. **YAML 파일 예시:**
    
    YAML 파일은 각 CSV 파일의 경로와 가중치 정보를 포함하고 있습니다. 아래는 YAML 파일의 예시입니다:
    ```yaml
    csv_files:
    - path: "../results/hj_pred1.csv"
    weight: 1
    - path: "../results/jw_pred2.csv"
    weight: 1
    - path: "../results/sy_pred.csv"
    weight: 1
    - path: "../results/sm_pred.csv"
    weight: 1
    - path: "../results/JE_pred.csv"
    weight: 0.5
    ```
## 😊 팀 구성원
<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://github.com/Heukma"><img src="https://avatars.githubusercontent.com/u/77618270?v=4" width="100px;" alt=""/><br /><sub><b>성효제</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/gagoory7"><img src="https://avatars.githubusercontent.com/u/163074222?v=4" width="100px;" alt=""/><br /><sub><b>백상민</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/Timeisfast"><img src="https://avatars.githubusercontent.com/u/120894109?v=4" width="100px;" alt=""/><br /><sub><b>김성윤</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/annakong23"><img src="https://avatars.githubusercontent.com/u/102771961?v=4" width="100px;" alt=""/><br /><sub><b>공지원</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/kimjueun028"><img src="https://avatars.githubusercontent.com/u/92249116?v=4" width="100px;" alt=""/><br /><sub><b>김주은</b></sub><br />
    </td>
    </td>
        <td align="center"><a href="https://github.com/zip-sa"><img src="https://avatars.githubusercontent.com/u/49730616?v=4" width="100px;" alt=""/><br /><sub><b>박승우</b></sub><br />
    </td>
  </tr>
</table>
</div>

<br />
