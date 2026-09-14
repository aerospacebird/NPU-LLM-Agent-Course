# NPU 기반 LLM Agent 서비스 구축 전문가 양성
## 1일 ~ 11일 상세 강의 교재 + 전체 코드

**기준 시간표**: 퓨리오사AI 훈련시간표 (80일·640시간)  
**운영시간**: 09:00 ~ 17:50 (1일 8시간)  
**장소**: 판교 한컴 AX LAB  
**강사**: 윤영선

---

# 1일 (08-31 월) – OT + Python 기초 + TensorFlow/Keras 실습 준비

## 학습 목표
- 과정 전체 로드맵과 목표 이해
- Python 가상환경 및 TensorFlow 2.x 설치/실행 확인
- Sequential API로 가장 간단한 예측 모델 생성

## 핵심 이론
- TensorFlow 2.x는 **Eager Execution**이 기본 (즉시 실행)
- `tf.keras.Sequential` : 층을 순차적으로 쌓는 가장 간단한 모델 구성 방식
- `Dense` Layer : Fully Connected Layer (모든 입력이 모든 출력에 연결)
- `model.summary()` : 모델 구조와 파라미터 수 확인
- `model.compile()` : 옵티마이저, 손실함수, 평가지표 설정

## 상세 설명
1. 아나콘다 또는 venv로 가상환경 생성 권장
2. `pip install tensorflow` (또는 tensorflow-gpu)
3. GPU 사용 가능 여부 확인: `tf.config.list_physical_devices('GPU')`
4. Sequential 모델은 입력 → Hidden → 출력 순으로 층을 추가

## 실습 코드 (전체)
```python
# ===== 1일 실습 코드 =====
import sys
print("Python version:", sys.version)

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# 가장 간단한 Sequential 모델
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,), name='hidden1'),
    tf.keras.layers.Dense(32, activation='relu', name='hidden2'),
    tf.keras.layers.Dense(1, name='output')
])

model.summary()

# 컴파일
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
print("모델 컴파일 완료")

# 간단한 더미 데이터로 fit 테스트
import numpy as np
X_dummy = np.random.rand(100, 10).astype(np.float32)
y_dummy = np.random.rand(100, 1).astype(np.float32)
history = model.fit(X_dummy, y_dummy, epochs=5, batch_size=16, verbose=1)
print("학습 테스트 완료")
```

## 체크포인트
- [ ] TensorFlow 정상 import 및 버전 확인
- [ ] GPU/CPU 환경 파악
- [ ] model.summary() 출력 확인
- [ ] 더미 데이터로 fit 성공

---

# 2일 (09-01 화) – 입력/출력 데이터 구조 + shape 확인

## 학습 목표
- 배열·행렬 기반 학습 데이터 구성 이해
- 모델 입출력 shape 확인 및 실습

## 핵심 이론
- 딥러닝에서 데이터는 주로 **NumPy ndarray** 또는 Tensor로 표현
- shape 표기법: `(samples, features)` 또는 `(batch, height, width, channels)`
- `model.input_shape`, `model.output_shape`로 확인 가능
- `model.predict()` 결과 shape도 반드시 확인

## 실습 코드
```python
# ===== 2일 실습 코드 =====
import numpy as np
import tensorflow as tf

# 가상의 데이터 생성
np.random.seed(42)
X = np.random.rand(1000, 8).astype(np.float32)   # (samples, features)
y = np.random.rand(1000, 1).astype(np.float32)

print("X shape:", X.shape)          # (1000, 8)
print("y shape:", y.shape)          # (1000, 1)
print("X dtype:", X.dtype)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

print("\n모델 입력 shape:", model.input_shape)   # (None, 8)
print("모델 출력 shape:", model.output_shape)   # (None, 1)

# 예측 shape 확인
pred = model.predict(X[:5], verbose=0)
print("예측 결과 shape:", pred.shape)          # (5, 1)
print("예측 값 예시:\n", pred)
```

## 체크포인트
- [ ] X, y shape 정확히 이해
- [ ] model.input_shape / output_shape 확인
- [ ] predict 결과 shape 일치 확인

---

# 3일 (09-02 수) – 은닉층·노드 수 조정 + batch_size / epochs

## 학습 목표
- 다층 퍼셉트론(MLP) 구조 이해
- 은닉층 깊이와 너비가 성능에 미치는 영향 실험
- batch_size와 epochs에 따른 학습 결과 비교

## 핵심 이론
- **은닉층 깊이(Depth)** ↑ → 표현력 증가, 과적합·학습 난이도 증가
- **은닉층 너비(Width)** ↑ → 한 층의 표현력 증가
- `batch_size`: 한 번에 학습에 사용하는 샘플 수 (작을수록 노이즈 많음, 클수록 안정적)
- `epochs`: 전체 데이터를 몇 번 반복 학습할지

## 실습 코드
```python
# ===== 3일 실습 코드 =====
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

X, y = make_regression(n_samples=2000, n_features=10, noise=0.1, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hidden_units=[64, 32]):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_units[0], activation='relu', input_shape=(10,)))
    for units in hidden_units[1:]:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 실험 1: 기본 구조
model1 = build_model([64, 32])
hist1 = model1.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# 실험 2: 더 넓은 네트워크
model2 = build_model([128, 64, 32])
hist2 = model2.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# 실험 3: batch_size 변경
model3 = build_model([64, 32])
hist3 = model3.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2, verbose=0)

print("Model1 (64-32) final val_loss:", hist1.history['val_loss'][-1])
print("Model2 (128-64-32) final val_loss:", hist2.history['val_loss'][-1])
print("Model3 (batch=128) final val_loss:", hist3.history['val_loss'][-1])

# 학습 곡선 비교
plt.figure(figsize=(10, 4))
plt.plot(hist1.history['val_loss'], label='64-32')
plt.plot(hist2.history['val_loss'], label='128-64-32')
plt.plot(hist3.history['val_loss'], label='batch=128')
plt.legend()
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Val Loss')
plt.show()
```

## 체크포인트
- [ ] 은닉층 구조 변경 실험
- [ ] batch_size / epochs 영향 관찰
- [ ] 학습 곡선 시각화

---

# 4일 (09-03 목) – train/test 분리 + 회귀 예측 시각화

## 학습 목표
- train/test 데이터 분리의 중요성 이해
- 회귀 모델 학습 후 예측값 시각화 (산점도 + 예측선)

## 핵심 이론
- `train_test_split`: 과적합 방지를 위해 반드시 분리
- 회귀에서는 예측값과 실제값을 산점도로 비교하는 것이 직관적
- 이상적인 경우 예측선이 y=x 직선에 가까움

## 실습 코드
```python
# ===== 4일 실습 코드 =====
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.random.set_seed(42)
X, y = make_regression(n_samples=1000, n_features=1, noise=15, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

y_pred = model.predict(X_test, verbose=0).flatten()

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, label='Actual', alpha=0.6, s=20)
plt.scatter(X_test, y_pred, label='Predicted', alpha=0.6, s=20)
# 정렬해서 예측선 그리기
sorted_idx = np.argsort(X_test.flatten())
plt.plot(X_test.flatten()[sorted_idx], y_pred[sorted_idx], 'r-', linewidth=2, label='Prediction Line')
plt.legend()
plt.title('Regression Prediction Visualization')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

## 체크포인트
- [ ] train/test 분리 적용
- [ ] 산점도 + 예측선 시각화 성공

---

# 5일 (09-04 금) – 회귀 성능평가 (R², RMSE) + 실데이터

## 학습 목표
- R², RMSE, MAE 지표 이해 및 계산
- Boston / California / Diabetes 데이터로 회귀 모델 구현

## 핵심 이론
- **R² (결정계수)**: 1에 가까울수록 좋음 (설명력)
- **RMSE**: 예측 오차의 제곱근 평균 (실제 단위와 동일)
- **MAE**: 절대 오차 평균
- 실데이터에서는 스케일링이 거의 필수

## 실습 코드
```python
# ===== 5일 실습 코드 =====
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

def evaluate_regression(X, y, name="Dataset"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    y_pred = model.predict(X_test, verbose=0).flatten()
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"[{name}] R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return model

# Diabetes
data = load_diabetes()
evaluate_regression(data.data, data.target, "Diabetes")

# California Housing
housing = fetch_california_housing()
evaluate_regression(housing.data, housing.target, "California Housing")
```

## 체크포인트
- [ ] R², RMSE, MAE 계산 및 해석
- [ ] 두 개 이상 실데이터셋 실험

---

# 6일 (09-07 월) – validation_split + 과적합 학습곡선

## 학습 목표
- validation_split으로 검증 데이터 자동 분리
- 학습 곡선으로 과적합 진단
- 학습 시간 측정

## 핵심 이론
- **과적합(Overfitting)**: train loss는 계속 감소하는데 val loss가 증가하는 현상
- 학습 곡선(Learning Curve)으로 시각적 진단
- `time` 모듈로 학습 시간 측정

## 실습 코드
```python
# ===== 6일 실습 코드 =====
import time
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

tf.random.set_seed(42)
data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

start = time.time()
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)
elapsed = time.time() - start
print(f"학습 소요 시간: {elapsed:.2f}초")

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Learning Curve (Overfitting Diagnosis)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

## 체크포인트
- [ ] validation_split 적용
- [ ] 학습 곡선으로 과적합 여부 판단
- [ ] 학습 시간 측정

---

# 7일 (09-08 화) – EarlyStopping + 이진분류

## 학습 목표
- EarlyStopping으로 과적합 방지
- 이진분류 모델 구현 (sigmoid + binary_crossentropy)
- 혼동행렬(Confusion Matrix) 기반 성능 평가

## 핵심 이론
- **sigmoid**: 출력을 0~1 확률로 변환
- **binary_crossentropy**: 이진분류 손실함수
- **EarlyStopping**: val_loss가 더 이상 개선되지 않으면 학습 중단
- `restore_best_weights=True` 권장

## 실습 코드
```python
# ===== 7일 실습 코드 =====
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

y_pred_proba = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## 체크포인트
- [ ] EarlyStopping 적용 및 조기 종료 확인
- [ ] 혼동행렬 + classification_report 출력

---

# 8일 (09-09 수) – 다중분류 (softmax + One-Hot + categorical_crossentropy)

## 학습 목표
- 다중분류 모델 구현
- softmax, One-Hot Encoding, categorical_crossentropy 이해
- Iris / Wine / Digits 데이터 실습

## 핵심 이론
- **softmax**: 여러 클래스에 대한 확률 분포 출력 (합=1)
- **One-Hot Encoding**: 정수 레이블 → [0,0,1,0,...] 형태
- `categorical_crossentropy` vs `sparse_categorical_crossentropy` 차이 이해

## 실습 코드
```python
# ===== 8일 실습 코드 =====
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

def multi_class_experiment(data, name):
    X, y = data.data, data.target
    n_classes = len(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-Hot Encoding
    y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, n_classes)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=100, batch_size=16, validation_split=0.2, verbose=0)
    
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"[{name}] Test Accuracy: {acc:.4f}")
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred))
    return model

multi_class_experiment(load_iris(), "Iris")
multi_class_experiment(load_wine(), "Wine")
# Digits는 데이터가 크므로 필요 시 실행
# multi_class_experiment(load_digits(), "Digits")
```

## 체크포인트
- [ ] One-Hot Encoding 적용
- [ ] softmax + categorical_crossentropy 사용
- [ ] 최소 2개 데이터셋 실험

---

# 9일 (09-10 목) – 데이터 스케일링 (MinMaxScaler vs StandardScaler)

## 학습 목표
- MinMaxScaler와 StandardScaler 차이 이해
- 스케일링 전후 회귀·분류 성능 비교

## 핵심 이론
- **StandardScaler**: 평균 0, 표준편차 1 (정규분포 가정에 유리)
- **MinMaxScaler**: 0~1 범위로 변환 (신경망 입력에 자주 사용)
- 스케일링은 **train에 fit → test에 transform** (데이터 누수 방지)

## 실습 코드
```python
# ===== 9일 실습 코드 =====
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)
data = load_diabetes()
X, y = data.data, data.target

def run_with_scaler(scaler_cls, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if scaler_cls is None:
        X_train_s, X_test_s = X_train, X_test
    else:
        scaler = scaler_cls()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_s.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_s, y_train, epochs=80, batch_size=32, verbose=0)
    
    y_pred = model.predict(X_test_s, verbose=0).flatten()
    r2 = r2_score(y_test, y_pred)
    print(f"[{name}] R² = {r2:.4f}")
    return r2

print("스케일링 비교 실험")
run_with_scaler(None, "No Scaling")
run_with_scaler(StandardScaler, "StandardScaler")
run_with_scaler(MinMaxScaler, "MinMaxScaler")
```

## 체크포인트
- [ ] 스케일링 전후 성능 차이 확인
- [ ] fit/transform 올바른 사용

---

# 10일 (09-11 금) – 모델 저장/불러오기 + ModelCheckpoint + 1~2주차 종합실습

## 학습 목표
- 모델 전체 저장 / weights만 저장
- ModelCheckpoint로 최고 성능 모델 자동 저장
- 1~2주차 내용 종합 실습

## 핵심 이론
- `model.save('xxx.keras')` : 구조+가중치+옵티마이저 상태 모두 저장 (권장)
- `model.save_weights()` : 가중치만 저장
- `tf.keras.models.load_model()` 로 불러오기
- ModelCheckpoint: 검증 성능이 개선될 때만 저장

## 실습 코드
```python
# ===== 10일 실습 코드 =====
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np

tf.random.set_seed(42)
data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# ModelCheckpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_diabetes.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint, early],
    verbose=0
)

# 전체 모델 저장
model.save('diabetes_full.keras')

# weights만 저장
model.save_weights('diabetes_weights.weights.h5')

# 불러오기 테스트
loaded = tf.keras.models.load_model('best_diabetes.keras')
y_pred = loaded.predict(X_test, verbose=0).flatten()
print("Loaded model R²:", r2_score(y_test, y_pred))
print("모델 저장/불러오기 성공!")
```

## 체크포인트
- [ ] .keras 파일 저장 및 불러오기 성공
- [ ] ModelCheckpoint로 best 모델 저장 확인
- [ ] 1~2주차 종합 실습 완료

---

# 11일 (09-14 월) – PCA 기본 개념 + 차원축소

## 학습 목표
- PCA(Principal Component Analysis) 기본 개념 이해
- explained variance ratio 분석
- MNIST 및 주요 데이터셋에 PCA 적용

## 핵심 이론
- PCA: 고차원 데이터를 분산이 큰 방향으로 투영하여 차원 축소
- **explained_variance_ratio_**: 각 주성분이 설명하는 분산 비율
- 누적 설명 분산이 95% 이상이 되는 성분 개수 선택 권장
- 차원 축소 후 모델 학습 속도·성능 비교

## 실습 코드
```python
# ===== 11일 실습 코드 =====
from sklearn.datasets import load_digits, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ----- 1. Digits 데이터로 PCA 시각화 -----
digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)  # 95% 분산 설명
X_pca = pca.fit_transform(X_scaled)

print(f"원본 차원: {X.shape[1]}")
print(f"PCA 후 차원: {X_pca.shape[1]}")
print(f"설명된 분산 비율 합: {pca.explained_variance_ratio_.sum():.4f}")

# 누적 설명 분산 그래프
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance Ratio')
plt.grid(True)
plt.show()

# ----- 2. PCA 적용 전후 분류 성능 비교 -----
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 원본
model_orig = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_orig.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_orig.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
acc_orig = model_orig.evaluate(X_test, y_test, verbose=0)[1]

# PCA 적용
pca = PCA(n_components=30)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model_pca = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_pca.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_pca.fit(X_train_pca, y_train, epochs=30, batch_size=32, verbose=0)
acc_pca = model_pca.evaluate(X_test_pca, y_test, verbose=0)[1]

print(f"원본 Accuracy: {acc_orig:.4f}")
print(f"PCA(30) Accuracy: {acc_pca:.4f}")
```

## 체크포인트
- [ ] explained_variance_ratio_ 이해 및 시각화
- [ ] PCA 전후 성능 비교
- [ ] 적절한 주성분 개수 선택

---

# 1~11일 종합 정리

| 일자 | 핵심 키워드 | 주요 코드 요소 |
|------|-------------|----------------|
| 1 | Sequential, Dense | model.summary(), compile |
| 2 | shape | input_shape, output_shape |
| 3 | MLP, batch_size, epochs | 은닉층 실험 |
| 4 | train/test split | 시각화 |
| 5 | R², RMSE | 실데이터 회귀 |
| 6 | validation_split, 과적합 | 학습 곡선 |
| 7 | EarlyStopping, 이진분류 | 혼동행렬 |
| 8 | softmax, One-Hot | 다중분류 |
| 9 | StandardScaler, MinMaxScaler | 스케일링 비교 |
| 10 | save, ModelCheckpoint | 모델 저장/불러오기 |
| 11 | PCA | explained_variance_ratio |

이 문서를 기반으로 각 일자 Jupyter Notebook을 실행하시면 됩니다.
