#!/usr/bin/env python
# coding: utf-8

# ## 기출복원 5회

# ### (작업형1) 

# #### 1.아래는 공공데이터 포털에서 제공하는 폐기물관리법」에 따른 지방자치단체별생활쓰레기 및 음식물쓰레기 종량제 봉투 가격에 대한 정보와 관련된 데이터의 일부이다. 전국의 2L 음식물쓰레기 규격봉투의 평균 가격을 계산하여라.(단, 평균 계산 시가격이 0원인 경우는 제외하고 결과는 소숫점을 버리고 정수로 출력할 것)

# In[1]:


#### 기출복원 05회 01 Solution
import pandas as pd
exam1 = pd.read_csv('data/기출복원/05회/전국_종량제봉투_가격표준.csv', encoding = 'cp949')

##### 음식물쓰레기 규격봉투(종량제 봉투 종류가 '규격봉투'이고 용도가 '음식물쓰레기')인 조건
cond1 = (exam1['종량제봉투종류']  == '규격봉투') & ( exam1['종량제봉투용도']  == '음식물쓰레기')

##### 2L 음식물쓰레기 규격봉투의 평균 가격
# 가격
price = exam1.loc[cond1, '2L가격'].copy()

# 평균(0원 제외)
avg_price = price[price > 0].mean()

##### 결과 저장
result = avg_price.astype('int')

##### 결과 출력
print(result)


# --- 

# #### 2. 주어진 데이터를 통해 BMI를 계산한 후 정상 체중에 속하는 인원과 과체중에 속하는 인원의 차이에 대한 절대값을 구하여라.(단, 출력시 정수로 출력할 것) 여기서 BMI는 몸무게/키의 제곱(몸무게의 단위는 kg, 키의 단위는 m)으로 계산되며, 아래와 같은 기준으로 분류된다
# * 18.5 미만이면 저체중
# * 18.5 이상이고 23.0미만이면 정상체중
# * 23.0 이상이고 25.0미만이면 과체중
# * 25.0 이상이면 비만

# In[2]:


#### 기출복원 05회 02 Solution
import pandas as pd
import numpy as np
exam2 = pd.read_csv('data/기출복원/05회/bmi.csv')

#### BMI 계산
# Height 컬럼과 Weight 컬럼 각각 할당
Height = exam2['Height']
Weight = exam2['Weight'] 

# BMI 계산
bmi = Weight/(Height/100)**2

#### 정상 체중에 속하는 인원과 과체중에 속하는 인원의 차이에 대한 절대값 계산
# 정상 체중인 사람의 수
a = sum((bmi >= 18.5) & (bmi < 23.0))

# 과체중인 사람의 수
b = sum((bmi >= 23.0) & (bmi < 25.0))

#### 차이에 대한 절대값 결과 저장
result = abs(a - b)

##### 결과 출력
print(result)


# --- 

# #### 3. 다음은 학교알리미에서 제공하는 공개용데이터의 일부로 부산광역시 교육청 산하의 초등학교의 전출입 현황이다. 순전입학생의 수가 가장 많은 순전입학교의 전체학생수를 구하라. 여기서 순전입학생의 수는 총 전입학생의 수 - 총 전출학생의 수로 계산되며, 순전입학교는 총 전입학생의 수가 총 전출학생의 수보다 많은 학교를 의미한다.

# In[3]:


#### 기출복원 05회 03 Solution
import pandas as pd
exam3 = pd.read_csv('data/기출복원/05회/2022_부산초등학교_전출입학생현황.csv', encoding = 'cp949')

##### 순전입학생 수가 가장 많은 순전입학교
# 순전입학생수 계산
exam3['순전입학생수'] = exam3['전입학생수합계'] - exam3['전출학생수합계']

# 순전입학교는 총 전입학생의 수가 총 전출학생수보다 많은 학교
# 즉, 순전입학생수(총 전입학생의 수 - 총 전출학생의 수) > 0 인 학교가 됨
# 순전입학교 = exam3[exam3['순전입학생수'] > 0].copy()를 통해 인덱싱한 후 구해도 되지만
# 순전입학교가 아닌 경우는 어차피 순전입학생수가 음수(0포함)이기 때문에 최댓값의 결과가 바뀌지 않음
# 따라서 별도의 인덱싱 없이 바로 최댓값을 구하면 됨

# 순전입학생 수가 가장 많은 경우의 행 인덱스 번호
idx_max = exam3['순전입학생수'].argmax()

#### 최종 결과 저장
result = exam3['전체학생수합계'][idx_max]

##### 결과 출력
print(result)


# ### (작업형2) 

# ---

# 1. 아래는 중고 포드(Ford) 자동차의 가격 예측을 위한 데이터의 일부이다.
# 13,470대에 대한 학습용 데이터를 이용하여 가격 예측 모형을 만든 후 이를 평가용 데이터에 적용하여 얻은 4,490대 예측 가격을 다음과 같은 형식의 CSV 파일로 생성하시오.(제출한 모델의 성능은 RMSE 평가지표에 따라 채점)

# In[4]:


#### 기출복원 05회차 작업형2 Solution

##### STEP1. 데이터셋 불러오기
import pandas as pd
train = pd.read_csv('data/기출복원/05회/carprice_train.csv')
test = pd.read_csv('data/기출복원/05회/carprice_test.csv')

# train 데이터를 X_train과 y_train으로 분할
y_train = train['price'].copy()
X_train = train.drop('price', axis = 1)

# test도 통일을 위해 X_test로 할당
X_test = test


# In[5]:


##### STEP2. 데이터셋 확인하기
###### STEP2-1. 데이터셋 일부 확인
print(X_train.head())
print(X_test.head())
print(y_train.head())


# In[6]:


###### STEP2-2. 데이터셋 요약 정보 확인
print(X_train.info())
print(X_test.info())
print(y_train.info())


# In[7]:


###### STEP2-3. 기초통계량 확인
print(X_train.describe())
print(X_test.describe())
print(y_train.describe())


# In[8]:


##### STEP3. 데이터셋 전처리
###### STEP3-1. 불필요한 컬럼 삭제
# 다른 문제들과 달리 고유번호는 없음


# In[9]:


###### STEP3-2. 결측치 처리
# 결측치 확인
X_train.isna().sum()


# In[10]:


X_test.isna().sum()


# In[11]:


###### STEP3-3. 카테고리형 컬럼 전처리
# 별도 과정이 없으므로 생략


# In[12]:


###### STEP3-4. 수치형 컬럼 전처리
####### year 컬럼
# 제작 연도로 범주형의 의미를 가짐, object형으로 변경
X_train['year'] = X_train['year'].astype('object')
X_test['year'] = X_test['year'].astype('object')


# In[13]:


###### STEP3-5. 데이터 분할
from sklearn.model_selection import train_test_split

# X_train과 y_train을 학습용(X_TRAIN, y_TRAIN)과 검증용(X_VAL, y_VAL)로 분할
X_TRAIN, X_VAL, y_TRAIN, y_VAL = train_test_split(X_train, y_train, random_state = 307, test_size = 0.2)

# 분할 후 shape 확인
print(X_TRAIN.shape)
print(X_VAL.shape)
print(y_TRAIN.shape)
print(y_VAL.shape)


# In[14]:


###### STEP3-6. 인코딩
# 카테고리형 컬럼에 대하여 원-핫 인코딩 수행
from sklearn.preprocessing import OneHotEncoder

# 인코딩할 카테고리형 컬럼만 별도 저장
X_TRAIN_category = X_TRAIN.select_dtypes('object').copy()
X_VAL_category = X_VAL.select_dtypes('object').copy()
X_TEST_category = X_test.select_dtypes('object').copy()

# 원-핫 인코딩
enc = OneHotEncoder(sparse = False).fit(X_TRAIN_category)
X_TRAIN_OH = enc.transform(X_TRAIN_category)
X_VAL_OH = enc.transform(X_VAL_category)
X_TEST_OH = enc.transform(X_TEST_category)


# In[15]:


###### STEP3-7. 스케일링
from sklearn.preprocessing import StandardScaler

# 스케일링할 컬럼만 별도 저장
# .select_dtypes() 메소드의 exclude 옵션은 해당 dtype을 제외한 모든 dtype을 추출할 때 사용
X_TRAIN_conti = X_TRAIN.select_dtypes(exclude = 'object').copy()
X_VAL_conti = X_VAL.select_dtypes(exclude = 'object').copy()
X_TEST_conti = X_test.select_dtypes(exclude = 'object').copy()

# TRAIN 데이터 기준으로 스케일링함
scale = StandardScaler().fit(X_TRAIN_conti)

# z-점수 표준화
X_TRAIN_STD = scale.transform(X_TRAIN_conti)
X_VAL_STD = scale.transform(X_VAL_conti)
X_TEST_STD = scale.transform(X_TEST_conti)


# In[16]:


###### STEP3-8. 입력 데이터셋 준비
import numpy as np

# 인코딩과 스케일링된 넘파이 배열 연결
X_TRAIN = np.concatenate([X_TRAIN_OH, X_TRAIN_STD], axis = 1)
X_VAL = np.concatenate([X_VAL_OH, X_VAL_STD], axis = 1)

# 1차원 넘파이 배열로 평탄화
y_TRAIN = y_TRAIN.values.ravel()
y_VAL = y_VAL.values.ravel()


# In[17]:


##### STEP4. 모델 학습
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

###### STEP4-1. random forest
rf = RandomForestRegressor(n_estimators = 100,
                           max_depth = 3,
                           min_samples_leaf = 10,
                           max_features = 30,
                           random_state = 2023)

model_rf = rf.fit(X_TRAIN, y_TRAIN)

###### STEP4-2. XGBoost
xgb = XGBRegressor(n_estimators = 100,
                   nthread = 5,
                   min_child_weight = 20,
                   gamma = 0.5,
                   objective = 'reg:squarederror',
                   use_label_encoder = False,
                   random_state = 2023)

model_xgb = xgb.fit(X_TRAIN, y_TRAIN, eval_metric = 'rmse')

###### STEP4-3. LightGBM
lgb = LGBMRegressor(max_depth = 8,
                    n_estimators = 100,
                    n_jobs = 30,
                    min_child_weight = 10,
                    learning_rate = 0.2,
                    objective = 'regression',
                    random_state = 2023)

model_lgb = lgb.fit(X_TRAIN, y_TRAIN)


# In[18]:


###### STEP4-4. 성능평가(기준:rmse)를 통한 모델 선정
from sklearn.metrics import mean_squared_error

# 검증용 데이터셋을 통한 예측
pred_rf = model_rf.predict(X_VAL)
pred_xgb = model_xgb.predict(X_VAL)
pred_lgb = model_lgb.predict(X_VAL)

# RMSE 계산
rmse_rf = mean_squared_error(y_VAL, pred_rf, squared = False)
print(rmse_rf)

rmse_xgb = mean_squared_error(y_VAL, pred_xgb, squared = False)
print(rmse_xgb)

rmse_lgb = mean_squared_error(y_VAL, pred_lgb, squared = False)
print(rmse_lgb)


# In[19]:


##### STEP5. 결과 제출하기
###### 실제 시험에서 답 제출시에는 성능이 가장 우수한 모형 하나만 구현!
X_TEST = np.concatenate([X_TEST_OH, X_TEST_STD], axis = 1)
y_pred = model_lgb.predict(X_TEST)

# 문제에서 요구하는 형태로 변환 필요
obj = {'pred' : y_pred}
result = pd.DataFrame(obj)

# 하위에 result.csv 이름으로 저장하기
result.to_csv("result.csv", index = False)


# In[20]:


##### STEP6. 채점 모델 평가(번외)
# 실제값
actual = pd.read_csv('data/기출복원/05회/carprice_y_test.csv')
actual = actual['price'].ravel()

# 채점 기준이 될 성과지표 값
mean_squared_error(actual, y_pred, squared = False)


# # (끝)
