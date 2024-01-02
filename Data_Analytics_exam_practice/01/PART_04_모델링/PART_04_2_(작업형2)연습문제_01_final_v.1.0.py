#!/usr/bin/env python
# coding: utf-8

# ## (작업형2) 연습문제

# #### 1. 아래는 가상 온라인 축구게임(FIFA)에 등장하는 축구 선수의 주요 특징과 신체 정보에 대한 일부 데이터이다.

# In[1]:


#### 연습문제1 Solution

##### STEP1. 데이터셋 불러오기
import pandas as pd
X_train = pd.read_csv('data/연습문제/FIFA_X_train.csv', encoding = 'cp949')
X_test = pd.read_csv('data/연습문제/FIFA_X_test.csv', encoding = 'cp949')
y_train = pd.read_csv('data/연습문제/FIFA_y_train.csv', encoding = 'cp949')


# In[2]:


##### STEP2. 데이터셋 확인하기
###### STEP2-1. 데이터셋 일부 확인
print(X_train.head())
print(X_test.head())
print(y_train.head())


# In[3]:


###### STEP2-2. 데이터셋 요약 정보 확인
print(X_train.info())
print(X_test.info())
print(y_train.info())


# In[5]:


###### STEP2-3. 기초통계량 확인
# 수치형 컬럼들의 기초통계 확인
print(X_train.describe())
print(X_test.describe())
print(y_train.describe())


# In[6]:


##### STEP3. 데이터셋 전처리
###### STEP3-1. 불필요한 컬럼 삭제
# ID 컬럼은 선수에 대한 고유 정보로 key 역할로 모델에는 불필요함
# 결과 제출 시에는 X_test의 ID 컬럼이 필요하기 때문에 별도 저장
ID = X_test['ID'].copy()

# 데이터들에서 ID 컬럼 삭제
X_train = X_train.drop(columns = 'ID')
X_test = X_test.drop(columns = 'ID')
y_train = y_train.drop(columns = 'ID')


# In[7]:


###### STEP3-2. 결측치 처리
# 결측치 확인
X_train.isna().sum()


# In[8]:


X_test.isna().sum()


# In[9]:


####### Position_Class 컬럼
# 선수 포지션을 의미하는 Position의 카테고리를 통합하는 과정에서 누락되었을 것
# 기존의 Position를 활용해 결측치를 대체
X_train['Position_Class'].value_counts() # 누락된 범주는 카운트되지 않음


# In[10]:


X_train['Position_Class'] = X_train['Position_Class'].fillna('unknown') # unknown으로 대체
X_train['Position_Class'].value_counts() # 확인


# In[11]:


# pandas.crosstab(index, colums)는 교차표를 생성하는 판다스 함수
# Postion 내 'CM', 'GK', 'LF', 'RDM', 'RWB'가 어느 Position_Class에도 속하지 않음
pd.crosstab(index = X_train['Position'], columns = X_train['Position_Class'])


# In[12]:


# X_train에 대해 누락된 카테고리 채우기
PC_train = X_train['Position_Class'].copy()

PC_train[X_train['Position'] == 'LF'] = 'Forward'
PC_train[X_train['Position'] == 'CM'] = 'Midfielder'
PC_train[X_train['Position'] == 'RDM'] = 'Defender'
PC_train[X_train['Position'] == 'RWB'] = 'Defender'
PC_train[X_train['Position'] == 'GK'] = 'GoalKeeper'

X_train['Position_Class'] = PC_train

# X_test에 대해 누락된 카테고리 채우기
PC_test = X_test['Position_Class'].copy()

PC_test[X_test['Position'] == 'LF'] = 'Forward'
PC_test[X_test['Position'] == 'CM'] = 'Midfielder'
PC_test[X_test['Position'] == 'RDM'] = 'Defender'
PC_test[X_test['Position'] == 'RWB'] = 'Defender'
PC_test[X_test['Position'] == 'GK'] = 'GoalKeeper'

X_test['Position_Class'] = PC_test


# In[13]:


# 재확인
pd.crosstab(index = X_train['Position'], columns = X_train['Position_Class'])


# In[14]:


# 반복문으로 하는 방법
lbl_pos = ['LF', 'CM', 'RDM', 'RWB', 'GK']
lbl_PC = ['Forward', 'Midfielder', 'Defender', 'Defender', 'GoalKeeper']

for r, s in zip(lbl_pos, lbl_PC):
    PC_train[X_train['Position'] == r] = s  
    PC_test[X_test['Position'] == r] = s


# In[15]:


# 완료 후 Position 컬럼을 삭제
X_train = X_train.drop(columns = 'Position')
X_test = X_test.drop(columns = 'Position')


# In[16]:


####### Height_cm 컬럼
# 단위가 인치와 피트인 문자열 Height를 단위 변환하는 과정에서 누락되었을 것
# 기존의 Height를 활용해 결측치를 대체

# X_train에 대해 누락된 값 채우기
Height_train = X_train['Height'].copy()
Height_cm_train = X_train['Height_cm'].copy()

# '를 기준으로 앞은 피트 * 30, 뒤는 인치 * 2.5한 후 합
# '를 기준으로 문자열을 분리한 후, expand = True를 통해 다른 열에 할당함
# 잘린 문자열은 수치형이랑 곱할 수 없으므로 astype() 메소드를 통해
# 각 열(시리즈)들의 dtype을 float64로 저장되도록 함
split_str_train = Height_train.str.split("'", expand = True).astype('float64')

# 결측치 대체
Height_cm_train = Height_cm_train.fillna(split_str_train[0] * 30 + split_str_train[1] * 2.5)
X_train['Height_cm'] = Height_cm_train

# X_test에 대해 누락된 값 채우기
Height_test = X_test['Height'].copy()
Height_cm_test = X_test['Height_cm'].copy()

# '를 기준으로 앞은 피트 * 30, 뒤는 인치 * 2.5한 후 합
# '를 기준으로 문자열을 분리한 후, expand = True를 통해 다른 열에 할당함
# 잘린 문자열은 수치형이랑 곱할 수 없으므로 astype() 메소드를 통해
# 각 열(시리즈)들의 dtype을 float64로 저장되도록 함
split_str_test = Height_test.str.split("'", expand = True).astype('float64')

# 결측치 대체
Height_cm_test = Height_cm_test.fillna(split_str_test[0] * 30 + split_str_test[1] * 2.5)
X_test['Height_cm'] = Height_cm_test

# 완료 후 Height 컬럼을 삭제
X_train = X_train.drop(columns = 'Height')
X_test = X_test.drop(columns = 'Height')


# In[17]:


####### Weight_lb 컬럼
# Weight_lb는 파운드 단위인 선수의 몸무게로 train에만 2% 결측이므로 행 삭제
# test에도 결측이 있을 경우, 시험에서는 삭제하면 안 되고 평균대치법과 같은 방법으로 대치해야함
# Weight_lb가 결측인 조건
cond_na = X_train['Weight_lb'].isna()

# y_train에 대해 X_train에 누락된 Weight_lb가 있는 행을 삭제함
y_train = y_train[~ cond_na]

# X_train에 대해 X_train에 누락된 Weight_lb가 있는 행을 삭제함
X_train = X_train[~ cond_na]

# 레코드 삭제후 행/열 길이 확인
print(y_train.shape, X_train.shape)


# In[18]:


###### STEP3-3. 카테고리형 컬럼 전처리
# 문자열(object) 컬럼들의 유일값 수 확인
print(X_train.select_dtypes('object').nunique())


# In[19]:


print(X_test.select_dtypes('object').nunique())


# In[20]:


####### Age컬럼
# 일부 선수의 나이가 일의 자리가 마스킹 되어있음
# Age_gp(연령대)인 카테고리형 파생변수 생성
X_train['Age_gp'] = X_train['Age'].str[0]
X_test['Age_gp'] = X_test['Age'].str[0]

# 완료 후 Age 컬럼을 삭제
X_train = X_train.drop('Age', axis = 1)
X_test = X_test.drop('Age', axis = 1)


# In[21]:


####### Club컬럼
# 현재 소속된 클럽으로, 예측에 불필요할 것으로 가정하고 컬럼을 삭제
X_train = X_train.drop(columns = 'Club')
X_test = X_test.drop(columns = 'Club')


# In[22]:


####### Preferred_Foot컬럼
# 선수가 주로 사용하는 발
print(X_train['Preferred_Foot'].value_counts())
print(X_test['Preferred_Foot'].value_counts())


# In[23]:


####### Work_Rate컬럼
# 공격 운동량/ 방어 운동량
# '/'를 기준으로 앞은 공격 운동량(WR_Attack), 뒤는 방어 운동량(WR_Defend) 컬럼으로 생성
# '/' 뒤에 공백하나가 있음으로 이에 대한 제거가 필요
# 그 이후 '/'를 기준으로 문자열을 분리한 후, expand = True를 통해 다른 열에 할당

# train
Work_Rate_train = X_train['Work_Rate'].copy()
Work_Rate_train = Work_Rate_train.str.replace(' ','') # 공백 제거

# '/'를 기준으로 문자열을 분리하여 파생변수 WR_Attack, WR_Defend 생성
X_train['WR_Attack'] = Work_Rate_train.str.split("/", expand = True)[0]
X_train['WR_Defend'] = Work_Rate_train.str.split("/", expand = True)[1]

# test
Work_Rate_test = X_test['Work_Rate'].copy()
Work_Rate_train = Work_Rate_train.str.replace(' ','') # 공백 제거

# '/'를 기준으로 문자열을 분리하여 파생변수 WR_Attack, WR_Defend 생성
X_test['WR_Attack'] = Work_Rate_test.str.split("/", expand = True)[0]
X_test['WR_Defend'] = Work_Rate_test.str.split("/", expand = True)[1]

# 완료 후 Work_Rate 컬럼을 삭제
X_train = X_train.drop(columns = 'Work_Rate')
X_test = X_test.drop(columns = 'Work_Rate')


# In[24]:


###### STEP3-4. 수치형 컬럼 전처리
####### Jersey_Number컬럼
# Jersey_Number는 선수의 등번호로 불필요한 컬럼으로 가정하고 삭제
# cf) 실제로 이 컬럼은 카테고리의 의미를 가짐을 주의해야함
X_train = X_train.drop('Jersey_Number', axis = 1)
X_test = X_test.drop('Jersey_Number', axis = 1)


# In[25]:


####### Contract_Valid_Until컬럼
# 계약 만료년도로 카테고리의 의미를 가짐
# 유일값 확인 결과, 2019~2026의 레이블을 가지고
# X_test에는 2026년은 없음
print(X_train['Contract_Valid_Until'].sort_values().unique())
print(X_test['Contract_Valid_Until'].sort_values().unique())

# CVU_gp컬럼으로 따로 저장
X_train['CVU_gp'] = X_train['Contract_Valid_Until'].astype('object') # dtype변환 
X_test['CVU_gp'] = X_test['Contract_Valid_Until'].astype('object') # dtype변환

# 완료 후 Contract_Valid_Until 컬럼을 삭제
X_train = X_train.drop('Contract_Valid_Until', axis = 1)
X_test = X_test.drop('Contract_Valid_Until', axis = 1)


# In[26]:


####### 수치형 컬럼 간 상관관계 확인
# 상관관계를 확인할 컬럼만
colnm_conti = ['Overall', 'Height_cm', 'Weight_lb', 'Release_Clause', 'Wage']
X_train[colnm_conti].corr()


# In[27]:


# Release_Clause 컬럼을 제외
X_train = X_train.drop('Release_Clause', axis = 1)
X_test = X_test.drop('Release_Clause', axis = 1)


# In[28]:


###### STEP3-5. 데이터 분할
from sklearn.model_selection import train_test_split
# X_train과 y_train을 학습용(X_TRAIN, y_TRAIN)과 검증용(X_VAL, y_VAL)로 분할
X_TRAIN, X_VAL, y_TRAIN, y_VAL = train_test_split(X_train, y_train, random_state = 1234, test_size = 0.3)

# 분할 후 shape 확인
print(X_TRAIN.shape)
print(X_VAL.shape)
print(y_TRAIN.shape)
print(y_VAL.shape)


# In[29]:


###### STEP3-6. 인코딩
# 카테고리형 컬럼에 대하여 원-핫 인코딩 수행
from sklearn.preprocessing import OneHotEncoder

# 인코딩할 카테고리형 컬럼만 별도 저장
X_TRAIN_category = X_TRAIN.select_dtypes('object').copy()
X_VAL_category = X_VAL.select_dtypes('object').copy()
X_TEST_category = X_test.select_dtypes('object').copy()

# Nationality의 유일 값 수가 데이터셋마다 다름
# handle_unknown = 'ignore'은 Train에 없는 레이블이 Test에 있더라도 이들을 모두 0이됨
enc = OneHotEncoder(handle_unknown = 'ignore', sparse = False).fit(X_TRAIN_category)

# 원-핫 인코딩
X_TRAIN_OH = enc.transform(X_TRAIN_category)
X_VAL_OH = enc.transform(X_VAL_category)
X_TEST_OH = enc.transform(X_TEST_category)


# In[30]:


###### STEP3-7. 스케일링
from sklearn.preprocessing import StandardScaler

# 스케일링할 컬럼만 별도 저장
colnm_conti = ['Overall', 'Height_cm', 'Weight_lb', 'Wage']
X_TRAIN_conti = X_TRAIN[colnm_conti].copy()
X_VAL_conti = X_VAL[colnm_conti].copy()
X_TEST_conti = X_test[colnm_conti].copy()

# TRAIN 데이터 기준으로 스케일링함
scale = StandardScaler().fit(X_TRAIN_conti)

# z-점수 표준화
X_TRAIN_STD = scale.transform(X_TRAIN_conti)
X_VAL_STD = scale.transform(X_VAL_conti)
X_TEST_STD = scale.transform(X_TEST_conti)


# In[31]:


###### STEP3-8. 입력 데이터셋 준비
import numpy as np

# 인코딩과 스케일링된 넘파이 ndarray 연결
X_TRAIN = np.concatenate([X_TRAIN_OH, X_TRAIN_STD], axis = 1)
X_VAL = np.concatenate([X_VAL_OH, X_VAL_STD], axis = 1)

# 1차원 넘파이 ndarray로 평탄화
y_TRAIN = y_TRAIN.values.ravel()
y_VAL = y_VAL.values.ravel()


# In[32]:


##### STEP4. 모델 학습
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor

###### STEP4-1. random forest
rf = RandomForestRegressor(n_estimators = 500,
                           max_depth = 3,
                           min_samples_leaf = 10,
                           max_features = 50,
                           random_state = 2022)
model_rf = rf.fit(X_TRAIN, y_TRAIN)

###### STEP4-2. Bagging
dtr = DecisionTreeRegressor(max_depth = 3, min_samples_leaf = 10)
bag = BaggingRegressor(base_estimator = dtr,
                       n_estimators = 500,
                       random_state = 2022)
model_bag = bag.fit(X_TRAIN, y_TRAIN)

###### STEP4-3. AdaBoost
dtr = DecisionTreeRegressor(max_depth = 3, min_samples_leaf = 10)
ada = AdaBoostRegressor(base_estimator = dtr,
                        n_estimators = 500,
                        learning_rate = 0.5,
                        random_state = 2022)
model_ada = ada.fit(X_TRAIN, y_TRAIN)


# In[33]:


###### STEP4-4. 성능평가(기준:RMSE)를 통한 모델 선정
from sklearn.metrics import mean_squared_error

# 검증용 데이터셋을 통한 예측
pred_rf = model_rf.predict(X_VAL)
pred_bag = model_bag.predict(X_VAL)
pred_ada = model_ada.predict(X_VAL)

# RMSE 계산
rmse_rf = mean_squared_error(y_VAL, pred_rf, squared = False)
print(rmse_rf)

rmse_bag = mean_squared_error(y_VAL, pred_bag, squared = False)
print(rmse_bag)

rmse_ada = mean_squared_error(y_VAL, pred_ada, squared = False)
print(rmse_ada)


# In[34]:


##### STEP5. 결과 제출하기
###### 실제 시험에서 답 제출시에는 성능이 가장 우수한 모형 하나만 구현!
X_TEST = np.concatenate([X_TEST_OH, X_TEST_STD], axis = 1)
y_pred = model_bag.predict(X_TEST)

# 문제에서 요구하는 형태로 변환 필요
obj = {'ID' : ID,
       'Purchase' : y_pred}
result = pd.DataFrame(obj)

# 하위에 12345.csv 이름으로 저장하기
result.to_csv("12345.csv", index = False)


# In[35]:


##### STEP6. 채점 모델 평가(번외)
# 실제값
actual = pd.read_csv('data/연습문제/FIFA_y_test.csv', encoding = 'cp949')
actual = actual['Value'].ravel()


# In[36]:


# 채점 기준이 될 성과지표 값
mean_squared_error(actual, y_pred, squared = False)


# ----

# #### 실제 시험에서 제출해야하는 답안

# In[37]:


#### 연습문제1 Solution

##### STEP1. 데이터셋 불러오기
import pandas as pd
X_train = pd.read_csv('data/연습문제/FIFA_X_train.csv', encoding = 'cp949')
X_test = pd.read_csv('data/연습문제/FIFA_X_test.csv', encoding = 'cp949')
y_train = pd.read_csv('data/연습문제/FIFA_y_train.csv', encoding = 'cp949')

##### STEP2. 데이터셋 확인하기
###### STEP2-1. 데이터셋 일부 확인
# print(X_train.head())
# print(X_test.head())
# print(y_train.head())

###### STEP2-2. 데이터셋 요약 정보 확인
# print(X_train.info())
# print(X_test.info())
# print(y_train.info())

###### STEP2-3. 기초통계량 확인
# 수치형 컬럼들의 기초통계 확인
# print(X_train.describe())
# print(X_test.describe())
# print(y_train.describe())

##### STEP3. 데이터셋 전처리
###### STEP3-1. 불필요한 컬럼 삭제
# ID 컬럼은 선수에 대한 고유 정보로 key 역할로 모델에는 불필요함
# 결과 제출 시에는 X_test의 ID 컬럼이 필요하기 때문에 별도 저장
ID = X_test['ID'].copy()

# 데이터들에서 ID 컬럼 삭제
X_train = X_train.drop(columns = 'ID')
X_test = X_test.drop(columns = 'ID')
y_train = y_train.drop(columns = 'ID')

###### STEP3-2. 결측치 처리
# 결측치 확인
X_train.isna().sum()
X_test.isna().sum()

####### Position_Class 컬럼
# 선수 포지션을 의미하는 Position의 카테고리를 통합하는 과정에서 누락되었을 것
# 기존의 Position를 활용해 결측치를 대체
# X_train['Position_Class'].value_counts() # 누락된 범주는 카운트되지 않음

# X_train['Position_Class'] = X_train['Position_Class'].fillna('unknown') # unknown으로 대체
# X_train['Position_Class'].value_counts() # 확인

# pandas.crosstab(index, colums)는 교차표를 생성하는 판다스 함수
# Postion 내 'CM', 'GK', 'LF', 'RDM', 'RWB'가 어느 Position_Class에도 속하지 않음
# pd.crosstab(index = X_train['Position'], columns = X_train['Position_Class'])

# X_train에 대해 누락된 카테고리 채우기
PC_train = X_train['Position_Class'].copy()

PC_train[X_train['Position'] == 'LF'] = 'Forward'
PC_train[X_train['Position'] == 'CM'] = 'Midfielder'
PC_train[X_train['Position'] == 'RDM'] = 'Defender'
PC_train[X_train['Position'] == 'RWB'] = 'Defender'
PC_train[X_train['Position'] == 'GK'] = 'GoalKeeper'

X_train['Position_Class'] = PC_train

# X_test에 대해 누락된 카테고리 채우기
PC_test = X_test['Position_Class'].copy()

PC_test[X_test['Position'] == 'LF'] = 'Forward'
PC_test[X_test['Position'] == 'CM'] = 'Midfielder'
PC_test[X_test['Position'] == 'RDM'] = 'Defender'
PC_test[X_test['Position'] == 'RWB'] = 'Defender'
PC_test[X_test['Position'] == 'GK'] = 'GoalKeeper'

X_test['Position_Class'] = PC_test

# 재확인
# pd.crosstab(index = X_train['Position'], columns = X_train['Position_Class'])

# 완료 후 Position 컬럼을 삭제
X_train = X_train.drop(columns = 'Position')
X_test = X_test.drop(columns = 'Position')

####### Height_cm 컬럼
# 단위가 인치와 피트인 문자열 Height를 단위 변환하는 과정에서 누락되었을 것
# 기존의 Height를 활용해 결측치를 대체

# X_train에 대해 누락된 값 채우기
Height_train = X_train['Height'].copy()
Height_cm_train = X_train['Height_cm'].copy()

# '를 기준으로 앞은 피트 * 30, 뒤는 인치 * 2.5한 후 합
# '를 기준으로 문자열을 분리한 후, expand = True를 통해 다른 열에 할당함
# 잘린 문자열은 수치형이랑 곱할 수 없으므로 astype() 메소드를 통해
# 각 열(시리즈)들의 dtype을 float64로 저장되도록 함
split_str_train = Height_train.str.split("'", expand = True).astype('float64')

# 결측치 대체
Height_cm_train = Height_cm_train.fillna(split_str_train[0] * 30 + split_str_train[1] * 2.5)
X_train['Height_cm'] = Height_cm_train

# X_test에 대해 누락된 값 채우기
Height_test = X_test['Height'].copy()
Height_cm_test = X_test['Height_cm'].copy()

# '를 기준으로 앞은 피트 * 30, 뒤는 인치 * 2.5한 후 합
# '를 기준으로 문자열을 분리한 후, expand = True를 통해 다른 열에 할당함
# 잘린 문자열은 수치형이랑 곱할 수 없으므로 astype() 메소드를 통해
# 각 열(시리즈)들의 dtype을 float64로 저장되도록 함
split_str_test = Height_test.str.split("'", expand = True).astype('float64')

# 결측치 대체
Height_cm_test = Height_cm_test.fillna(split_str_test[0] * 30 + split_str_test[1] * 2.5)
X_test['Height_cm'] = Height_cm_test

# 완료 후 Height 컬럼을 삭제
X_train = X_train.drop(columns = 'Height')
X_test = X_test.drop(columns = 'Height')

####### Weight_lb 컬럼
# Weight_lb는 파운드 단위인 선수의 몸무게로 train에만 2% 결측이므로 행 삭제
# test에도 결측이 있을 경우, 시험에서는 삭제하면 안 되고 평균대치법과 같은 방법으로 대치해야함
# Weight_lb가 결측인 조건
cond_na = X_train['Weight_lb'].isna()

# y_train에 대해 X_train에 누락된 Weight_lb가 있는 행을 삭제함
y_train = y_train[~ cond_na]

# X_train에 대해 X_train에 누락된 Weight_lb가 있는 행을 삭제함
X_train = X_train[~ cond_na]

# 레코드 삭제후 행/열 길이 확인
# print(y_train.shape, X_train.shape)

###### STEP3-3. 카테고리형 컬럼 전처리
# 문자열(object) 컬럼들의 유일값 수 확인
# .select_dtypes()은 원하는 dtype을 가진 데이터프레임만 추출 후 유일값 수 확인
# print(X_train.select_dtypes('object').nunique())
# print(X_test.select_dtypes('object').nunique())

####### Age컬럼
# 일부 선수의 나이가 일의 자리가 마스킹 되어있음
# Age_gp(연령대)인 카테고리형 파생변수 생성
X_train['Age_gp'] = X_train['Age'].str[0]
X_test['Age_gp'] = X_test['Age'].str[0]

# 완료 후 Age 컬럼을 삭제
X_train = X_train.drop('Age', axis = 1)
X_test = X_test.drop('Age', axis = 1)

####### Club컬럼
# 현재 소속된 클럽으로, 예측에 불필요할 것으로 가정하고 컬럼을 삭제
X_train = X_train.drop(columns = 'Club')
X_test = X_test.drop(columns = 'Club')

####### Preferred_Foot컬럼
# 선수가 주로 사용하는 발
# print(X_train['Preferred_Foot'].value_counts())
# print(X_test['Preferred_Foot'].value_counts())

####### Work_Rate컬럼
# 공격 운동량/ 방어 운동량
# '/'를 기준으로 앞은 공격 운동량(WR_Attack), 뒤는 방어 운동량(WR_Defend) 컬럼으로 생성
# '/' 뒤에 공백하나가 있음으로 이에 대한 제거가 필요
# 그 이후 '/'를 기준으로 문자열을 분리한 후, expand = True를 통해 다른 열에 할당

# train
Work_Rate_train = X_train['Work_Rate'].copy()
Work_Rate_train = Work_Rate_train.str.replace(' ','') # 공백 제거

# '/'를 기준으로 문자열을 분리하여 파생변수 WR_Attack, WR_Defend 생성
X_train['WR_Attack'] = Work_Rate_train.str.split("/", expand = True)[0]
X_train['WR_Defend'] = Work_Rate_train.str.split("/", expand = True)[1]

# test
Work_Rate_test = X_test['Work_Rate'].copy()
Work_Rate_train = Work_Rate_train.str.replace(' ','') # 공백 제거

# '/'를 기준으로 문자열을 분리하여 파생변수 WR_Attack, WR_Defend 생성
X_test['WR_Attack'] = Work_Rate_test.str.split("/", expand = True)[0]
X_test['WR_Defend'] = Work_Rate_test.str.split("/", expand = True)[1]

# 완료 후 Work_Rate 컬럼을 삭제
X_train = X_train.drop(columns = 'Work_Rate')
X_test = X_test.drop(columns = 'Work_Rate')

###### STEP3-4. 수치형 컬럼 전처리
####### Jersey_Number컬럼
# Jersey_Number는 선수의 등번호로 불필요한 컬럼으로 가정하고 삭제
# cf) 실제로 이 컬럼은 카테고리의 의미를 가짐을 주의해야함
X_train = X_train.drop('Jersey_Number', axis = 1)
X_test = X_test.drop('Jersey_Number', axis = 1)

####### Contract_Valid_Until컬럼
# 계약 만료년도로 카테고리의 의미를 가짐
# 유일값 확인 결과, 2019~2026의 레이블을 가지고
# X_test에는 2026년은 없음
# print(X_train['Contract_Valid_Until'].sort_values().unique())
# print(X_test['Contract_Valid_Until'].sort_values().unique())

# CVU_gp컬럼으로 따로 저장
X_train['CVU_gp'] = X_train['Contract_Valid_Until'].astype('object') # dtype변환 
X_test['CVU_gp'] = X_test['Contract_Valid_Until'].astype('object') # dtype변환

# 완료 후 Contract_Valid_Until 컬럼을 삭제
X_train = X_train.drop('Contract_Valid_Until', axis = 1)
X_test = X_test.drop('Contract_Valid_Until', axis = 1)

####### 수치형 컬럼 간 상관관계 확인
# 상관관계를 확인할 컬럼만
colnm_conti = ['Overall', 'Height_cm', 'Weight_lb', 'Release_Clause', 'Wage']
X_train[colnm_conti].corr()

# Release_Clause 컬럼을 제외
X_train = X_train.drop('Release_Clause', axis = 1)
X_test = X_test.drop('Release_Clause', axis = 1)

###### STEP3-5. 데이터 분할
from sklearn.model_selection import train_test_split
# X_train과 y_train을 학습용(X_TRAIN, y_TRAIN)과 검증용(X_VAL, y_VAL)로 분할
X_TRAIN, X_VAL, y_TRAIN, y_VAL = train_test_split(X_train, y_train, random_state = 1234, test_size = 0.3)

# 분할 후 shape 확인
# print(X_TRAIN.shape)
# print(X_VAL.shape)
# print(y_TRAIN.shape)
# print(y_VAL.shape)

###### STEP3-6. 인코딩
# 카테고리형 컬럼에 대하여 원-핫 인코딩 수행
from sklearn.preprocessing import OneHotEncoder

# 인코딩할 카테고리형 컬럼만 별도 저장
X_TRAIN_category = X_TRAIN.select_dtypes('object').copy()
# X_VAL_category = X_VAL.select_dtypes('object').copy()
X_TEST_category = X_test.select_dtypes('object').copy()

# Nationality의 유일 값 수가 데이터셋마다 다름
# handle_unknown = 'ignore'은 Train에 없는 레이블이 Test에 있더라도 이들을 모두 0이됨
enc = OneHotEncoder(handle_unknown = 'ignore', sparse = False).fit(X_TRAIN_category)

# 원-핫 인코딩
X_TRAIN_OH = enc.transform(X_TRAIN_category)
# X_VAL_OH = enc.transform(X_VAL_category)
X_TEST_OH = enc.transform(X_TEST_category)

###### STEP3-7. 스케일링
from sklearn.preprocessing import StandardScaler

# 스케일링할 컬럼만 별도 저장
colnm_conti = ['Overall', 'Height_cm', 'Weight_lb', 'Wage']
X_TRAIN_conti = X_TRAIN[colnm_conti].copy()
# X_VAL_conti = X_VAL[colnm_conti].copy()
X_TEST_conti = X_test[colnm_conti].copy()

# TRAIN 데이터 기준으로 스케일링함
scale = StandardScaler().fit(X_TRAIN_conti)

# z-점수 표준화
X_TRAIN_STD = scale.transform(X_TRAIN_conti)
# X_VAL_STD = scale.transform(X_VAL_conti)
X_TEST_STD = scale.transform(X_TEST_conti)

###### STEP3-8. 입력 데이터셋 준비
import numpy as np

# 인코딩과 스케일링된 넘파이배열 연결
X_TRAIN = np.concatenate([X_TRAIN_OH, X_TRAIN_STD], axis = 1)
# X_VAL = np.concatenate([X_VAL_OH, X_VAL_STD], axis = 1)

# 1차원 넘파이배열로 평탄화
y_TRAIN = y_TRAIN.values.ravel()
# y_VAL = y_VAL.values.ravel()

##### STEP4. 모델 학습
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor

###### STEP4-1. random forest
# rf = RandomForestRegressor(n_estimators = 500,
#                            max_depth = 3,
#                            min_samples_leaf = 10,
#                            max_features = 50,
#                            random_state = 2022)
# model_rf = rf.fit(X_TRAIN, y_TRAIN)

###### STEP4-2. Bagging
dtr = DecisionTreeRegressor(max_depth = 3, min_samples_leaf = 10)
bag = BaggingRegressor(base_estimator = dtr,
                       n_estimators = 500,
                       random_state = 2022)
model_bag = bag.fit(X_TRAIN, y_TRAIN)

###### STEP4-3. AdaBoost
# dtr = DecisionTreeRegressor(max_depth = 3, min_samples_leaf = 10)
# ada = AdaBoostRegressor(base_estimator = dtr,
#                         n_estimators = 500,
#                         learning_rate = 0.5,
#                         random_state = 2022)
# model_ada = ada.fit(X_TRAIN, y_TRAIN)

###### STEP4-4. 성능평가(기준:RMSE)를 통한 모델 선정
# from sklearn.metrics import mean_squared_error

# 검증용 데이터셋을 통한 예측
# pred_rf = model_rf.predict(X_VAL)
# pred_bag = model_bag.predict(X_VAL)
# pred_ada = model_ada.predict(X_VAL)

# RMSE 계산
# rmse_rf = mean_squared_error(y_VAL, pred_rf, squared = False)
# print(rmse_rf)

# rmse_bag = mean_squared_error(y_VAL, pred_bag, squared = False)
# print(rmse_bag)

# rmse_ada = mean_squared_error(y_VAL, pred_ada, squared = False)
# print(rmse_ada)

##### STEP5. 결과 제출하기
###### 실제 시험에서 답 제출시에는 성능이 가장 우수한 모형 하나만 구현!
X_TEST = np.concatenate([X_TEST_OH, X_TEST_STD], axis = 1)
y_pred = model_bag.predict(X_TEST)

# 문제에서 요구하는 형태로 변환 필요
obj = {'ID' : ID,
       'Purchase' : y_pred}
result = pd.DataFrame(obj)

# 하위에 12345.csv 이름으로 저장하기
result.to_csv("abc.csv", index = False)


# ## (끝)
