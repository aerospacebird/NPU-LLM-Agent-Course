# NPU 기반 LLM Agent 서비스 구축 전문가 양성
## 12일 ~ 18일 상세 강의 교재 + 전체 코드

**기준**: 퓨리오사AI 훈련시간표 | **강사**: 윤영선 | **장소**: 판교 한컴 AX LAB

---

# 12일 (09-15 화) – 로그 변환 + KFold 교차검증

## 학습 목표
- 로그 변환(`np.log1p`)으로 왜도(skewness) 감소 및 회귀 성능 개선
- KFold 교차검증 개념 이해
- 단일 train/test split vs KFold 성능 안정성 비교

## 핵심 이론
- 타깃 변수가 오른쪽으로 치우친 경우 로그 변환이 효과적
- KFold: 데이터를 K등분하여 K번 학습/검증 → 평균·표준편차로 성능 추정
- `cross_val_score`로 간편 평가

## 실습 포인트
- California Housing 분포 시각화
- 로그 변환 전후 Ridge 회귀 R²/RMSE 비교
- 5-Fold CV mean/std 출력

---

# 13일 (09-16 수) – all_estimators 모델 비교

## 학습 목표
- `sklearn.utils.all_estimators`로 분류/회귀 모델 일괄 벤치마크
- 데이터셋별 Top 모델 선정

## 핵심 이론
- type_filter='regressor' / 'classifier'
- 일부 모델은 특정 조건에서 실패 → try-except 처리 필요
- 빠른 후보 선정에 유용 (정밀 튜닝 전 단계)

---

# 14일 (09-17 목) – GridSearchCV / RandomizedSearchCV + XGBoost

## 학습 목표
- GridSearchCV(전수 탐색) vs RandomizedSearchCV(랜덤 샘플링) 차이
- XGBoost 주요 하이퍼파라미터 튜닝

## 핵심 파라미터
- max_depth, learning_rate, n_estimators, subsample, colsample_bytree

## 실습 포인트
- 회귀: GridSearchCV
- 분류: RandomizedSearchCV (n_iter=20)
- 소요 시간 및 Test 성능 비교

---

# 15일 (09-18 금) – SMOTE + KMeans + HalvingSearch

## 학습 목표
- 불균형 데이터 처리 (SMOTE, RandomOverSampler)
- KMeans 기초 (Elbow, Silhouette)
- HalvingGridSearch / HalvingRandomSearch

## 핵심 이론
- SMOTE: 소수 클래스 합성 샘플 생성
- HalvingSearch: 자원을 점진적으로 늘리며 후보를 줄여 빠른 탐색

---

# 16일 (09-21 월) – Feature Importance + 특성 선택

## 학습 목표
- Tree 기반 Feature Importance 시각화
- SelectFromModel, 상위 N개 특성 선택 후 성능 비교

## 실습 포인트
- RandomForest / XGBoost importance
- threshold='median' 또는 Top-K 선택
- 원본 vs 선택 후 Accuracy 비교

---

# 17일 (09-22 화) – Bayesian Optimization (Optuna)

## 학습 목표
- Bayesian Optimization 개념 (이전 결과 활용 지능적 탐색)
- Optuna로 XGBoost 회귀/분류 튜닝

## 핵심 이론
- Grid/Random보다 시행 횟수 대비 효율적
- suggest_int / suggest_float / log=True 활용

---

# 18일 (09-23 수) – Keras Optimizer + ReduceLROnPlateau

## 학습 목표
- Adam, RMSprop, SGD, SGD+Momentum 비교
- ReduceLROnPlateau로 학습률 동적 감소

## 핵심 이론
- Optimizer마다 수렴 속도와 안정성 차이
- val_loss 개선 없을 때 LR을 factor배 감소 (min_lr까지)

---

## 노트북 파일 목록
- 12_Log_Transform_KFold.ipynb
- 13_All_Estimators_Comparison.ipynb
- 14_Grid_Random_Search_XGBoost.ipynb
- 15_SMOTE_KMeans_HalvingSearch.ipynb
- 16_Feature_Importance_Selection.ipynb
- 17_Bayesian_Optimization.ipynb
- 18_Keras_Optimizer_ReduceLR.ipynb

각 노트북에 실행 가능한 전체 코드가 포함되어 있습니다.
