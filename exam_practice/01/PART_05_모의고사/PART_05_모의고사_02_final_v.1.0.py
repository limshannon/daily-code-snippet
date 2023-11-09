#!/usr/bin/env python
# coding: utf-8

# ## 모의고사 2회

# ### (작업형1) 

# #### 1. USArrests 데이터셋을 불러와 UrbanPop이 60이상인 지역 중 Murder와 Assault의 합 대비 Assault의 비율이 0.05이상인 레코드 수를 구하여라.

# In[23]:


#### 모의고사 02회 01 Solution
import pandas as pd
exam1 = pd.read_csv('data/모의고사/02회/USArrests.csv')

##### Murder와 Assault의 합 대비 Assault의 비율에 대한 컬럼 생성
# Murder와 Assault의 합
exam1['MA'] = exam1['Murder'] + exam1['Assault']

# Murder와 Assault의 합 대비 Assault의 비율
exam1['ratio'] = exam1['Assault']/exam1['MA']

##### UrbanPop이 60이상이고 ratio가 0.05이상인 경우
cond = (exam1['UrbanPop'] >= 60) & (exam1['ratio'] >= 0.05)

##### 결과를 result에 할당
result = exam1[cond].shape[0]

##### 결과 출력
print(result)


# --- 

# #### 2. swiss 데이터셋을 불러와 Fertility 컬럼에 대해서 내림차순으로 정렬한 후 정렬한 데이터를 기준으로 홀수번째 레코드들의 평균에서 짝수번째 레코드들의 평균을 뺀 값을 구하여라. (단, 첫번째 행에 있는 데이터를 1번으로 하고, 결과는 소수점 넷째 자리에서 반올림하여 표현)

# In[24]:


#### 모의고사 02회 02 Solution
import pandas as pd
exam2 = pd.read_csv('data/모의고사/02회/swiss.csv')

##### Fertility 컬럼에 대해서 내림차순으로 정렬
sort = exam2['Fertility'].sort_values(ascending = False, ignore_index = True)

##### 데이터의 홀수번째와 짝수번째 행 번호 생성
import numpy as np
idx = np.arange(1,48)

# 홀수
odd = (idx % 2 == 1)

# 짝수
even = (idx % 2 == 0)

# 차이
diff = sort[odd].mean() - sort[even].mean()

##### 결과를 result에 할당
result = round(diff, 3)

##### 결과 출력
print(result)


# --- 

# #### 3. CO2 데이터셋을 불러와 Type 컬럼이 Mississippi이면서 conc 컬럼에서 백의 자리 또는 일의 자리가 5인 경우 레코드들의 수를 구하여라.

# In[1]:


#### 모의고사 02회차 03 Solution
import pandas as pd
exam3 = pd.read_csv('data/모의고사/02회/CO2.csv')

##### case1. Type 컬럼이 Mississippi인 경우
# 'Mississi/ppi'와 'Mis/sissippi'이 섞여있음
# exam3['Type'].value_counts()

# '/'를 제거
exam3['Type'] = exam3['Type'].str.replace('/', '')

# Mississippi일 조건
case1 = (exam3['Type'] == 'Mississippi')

##### case2. conc 컬럼에서 백의 자리 또는 일의 자리가 5인 경우
# 백의 자리가 5인 경우
hundred = exam3['conc']//100  == 5

# 일의 자리가 5인 경우
one = exam3['conc'].astype('string').str.endswith('5')

# 두 조건을 만족하는 조건
case2 = hundred | one

##### 결과를 result에 할당
result = exam3[case1 & case2].shape[0]

##### 결과 출력
print(result)


# ---

# ### (작업형2) 

# ### 1. 아래는 블랙프라이데이 제품 구매자들의 구매 정보에 관련한 데이터의 일부이다. 주어진 데이터를 이용하여 예측 모형을 만들고 아래에 따라 CSV 파일을 생성하시오.

# In[26]:


#### 모의고사 01회차_작업형2 Solution

##### STEP1. 데이터셋+ 불러오기
import pandas as pd
X_train = pd.read_csv('data/모의고사/02회/BlackFriday_X_train.csv')
X_test = pd.read_csv('data/모의고사/02회/BlackFriday_X_test.csv')
y_train = pd.read_csv('data/모의고사/02회/BlackFriday_y_train.csv')


# In[27]:


##### STEP2. 데이터셋 확인하기
###### STEP2-1. 데이터셋 일부 확인
print(X_train.head())
print(X_test.head())
print(y_train.head())


# In[28]:


###### STEP2-2. 데이터셋 요약 정보 확인
print(X_train.info())
print(X_test.info())
print(y_train.info())


# In[29]:


###### STEP2-3. 기초통계량 확인
print(X_train.describe())
print(X_test.describe())
print(y_train.describe())


# In[30]:


##### STEP3. 데이터셋 전처리
###### STEP3-1. 불필요한 컬럼 삭제
# User_ID 컬럼은 구매자에 대한 고유 정보로 key 역할로 모델에는 불필요함
# 결과 제출 시에는 X_test의 ID 컬럼이 필요하기 때문에 별도 저장
User_ID = X_test['User_ID'].copy()

# Product_ID는 제품의 고유 ID로 마찬가지로 삭제함

# 데이터들에서 User_ID, Product_ID 컬럼 삭제
X_train = X_train.drop(columns = ['User_ID', 'Product_ID'])
X_test = X_test.drop(columns = ['User_ID', 'Product_ID'])
y_train = y_train.drop(columns = 'User_ID')


# In[31]:


###### STEP3-2. 결측치 처리
# 결측치 확인
X_train.isna().sum()


# In[32]:


X_test.isna().sum()


# In[33]:


####### Product_Category_2 컬럼(train 1205, test 848 결측)
# train은 레코드의 31%, test는 레코드의 33%가 결측이고 Product_Category_1의 하위 카테고리
# 컬럼을 삭제
X_train = X_train.drop('Product_Category_2', axis = 1)
X_test = X_test.drop('Product_Category_2', axis = 1)

####### Product_Category_3 컬럼(train 2687, test 1807 결측)
# train은 레코드의 69%, test는 레코드의 70%가 결측이고 Product_Category_1, 2의 하위 카테고리
# 컬럼을 삭제
X_train = X_train.drop('Product_Category_3', axis = 1)
X_test = X_test.drop('Product_Category_3', axis = 1)


# In[34]:


###### STEP3-3. 카테고리형 컬럼 전처리
# 문자열(object) 컬럼들의 유일값 수 확인
# 컬럼별 카테고리 확인 결과 큰 이상 없음
print(X_train.select_dtypes('object').nunique())
print(X_test.select_dtypes('object').nunique())


# In[35]:


###### STEP3-4. 수치형 컬럼 전처리
print(X_train.select_dtypes(exclude ='object'))
print(X_test.select_dtypes(exclude ='object'))

####### Occupation, Marital_Status, Product_Category_1컬럼
# 수치형으로 인식되지만 카테고리의 의미를 가지는 컬럼
# dtype 변경 후 각각 OCC_gp, Marital_gp, PC_gp에 할당
X_train['OCC_gp'] = X_train['Occupation'].astype('object')
X_test['OCC_gp'] = X_test['Occupation'].astype('object')

X_train['Marital_gp'] = X_train['Marital_Status'].astype('object')
X_test['Marital_gp'] = X_test['Marital_Status'].astype('object')

X_train['PC_gp'] = X_train['Product_Category_1'].astype('object')
X_test['PC_gp'] = X_test['Product_Category_1'].astype('object')

# 기존 컬럼 삭제
X_train = X_train.drop(['Occupation', 'Marital_Status', 'Product_Category_1'], axis = 1)
X_test = X_test.drop(['Occupation', 'Marital_Status', 'Product_Category_1'], axis = 1)


# In[36]:


###### STEP3-5. 데이터 분할
from sklearn.model_selection import train_test_split

# X_train과 y_train을 학습용(X_TRAIN, y_TRAIN)과 검증용(X_VAL, y_VAL)로 분할
X_TRAIN, X_VAL, y_TRAIN, y_VAL = train_test_split(X_train, y_train, random_state = 1234, test_size = 0.3)

# 분할 후 shape 확인
print(X_TRAIN.shape)
print(X_VAL.shape)
print(y_TRAIN.shape)
print(y_VAL.shape)


# In[37]:


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


# In[38]:


###### STEP3-7. 스케일링
# 스케일링할 컬럼 없음


# In[39]:


###### STEP3-8. 입력 데이터셋 준비
import numpy as np

# 인코딩과 스케일링된 넘파이 배열 연결
X_TRAIN = X_TRAIN_OH
X_VAL = X_VAL_OH

# 1차원 넘파이배열로 평탄화
y_TRAIN = y_TRAIN.values.ravel()
y_VAL = y_VAL.values.ravel()


# In[40]:


##### STEP4. 모델 학습
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

###### STEP4-1. random forest
rf = RandomForestRegressor(n_estimators = 500,
                           max_depth = 3,
                           min_samples_leaf = 10,
                           max_features = 2,
                           random_state = 2022)

model_rf = rf.fit(X_TRAIN, y_TRAIN)

###### STEP4-2. XGBoost
xgb = XGBRegressor(max_depth = 8,
                   n_estimators = 500,
                   nthread = 5,
                   min_child_weight = 20,
                   gamma = 0.5,
                   objective = 'reg:squarederror',
                   use_label_encoder = False,
                   random_state = 2022)

model_xgb = xgb.fit(X_TRAIN, y_TRAIN)

###### STEP4-3. LightGBM
lgb = LGBMRegressor(max_depth = 8,
                    n_estimators = 500,
                    n_jobs = 30,
                    min_child_weight = 10,
                    learning_rate = 0.2,
                    objective = 'regression',
                    random_state = 2022)

model_lgb = lgb.fit(X_TRAIN, y_TRAIN)


# In[41]:


###### STEP4-4. 성능평가(기준:MAE)를 통한 모델 선정
from sklearn.metrics import mean_absolute_error

# 검증용 데이터셋을 통한 예측
pred_rf = model_rf.predict(X_VAL)
pred_xgb = model_xgb.predict(X_VAL)
pred_lgb = model_lgb.predict(X_VAL)

# MAE 계산
mae_rf = mean_absolute_error(y_VAL, pred_rf)
print(mae_rf)

mae_xgb = mean_absolute_error(y_VAL, pred_xgb)
print(mae_xgb)

mae_lgb = mean_absolute_error(y_VAL, pred_lgb)
print(mae_lgb)


# In[42]:


##### STEP5. 결과 제출하기
###### 실제 시험에서 답 제출시에는 성능이 가장 우수한 모형 하나만 구현!
X_TEST = X_TEST_OH
y_pred = model_lgb.predict(X_TEST)

# 문제에서 요구하는 형태로 변환 필요
obj = {'User_ID' : User_ID,
       'Purchase' : y_pred}
result = pd.DataFrame(obj)

# 하위에 12345.csv 이름으로 저장하기
result.to_csv("12345.csv", index = False)


# In[43]:


##### STEP6. 채점 모델 평가(번외)
# 실제값
actual = pd.read_csv('data/모의고사/02회/BlackFriday_y_test.csv', encoding = 'cp949')
actual = actual['Purchase'].ravel()

# 채점 기준이 될 성과지표 값
mean_absolute_error(actual, y_pred)


# # (끝)
