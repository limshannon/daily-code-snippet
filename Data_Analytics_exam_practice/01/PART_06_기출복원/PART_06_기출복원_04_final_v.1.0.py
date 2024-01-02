#!/usr/bin/env python
# coding: utf-8

# ## 기출복원 4회

# ### (작업형1) 

# #### 1. Cars93  데이터셋을 불러와 Weight 컬럼에 대하여 아래의 과정을 수행하여라.

# (가) 제 1사분위수와 제 2사분위수를 구하기
# (나) 두 개의 차이의 절댓값을 구하기
# (다) 그 값의 소수점을 버리기

# In[17]:


#### 기출복원 04회 01 Solution
import pandas as pd
exam1 = pd.read_csv('data/기출복원/04회/Cars93.csv')

#### Weight 컬럼 별도 저장
Weight = exam1['Weight']

#### (가) 제 1사분위수와 제 2사분위수 구하기
q1 = Weight.quantile(.25)
q2 = Weight.quantile(.5)

#### (나) 두 개의 차이의 절댓값을 구하기
diff = abs(q1 - q2)

##### 결과를 result에 할당
# (다) 그 값에 소수점을 버리기
result = diff.astype('int')

##### 결과 출력
print(result)


# --- 

# #### 2. facebook_subset 데이터셋을 불러와 좋아요를 받은 전체 수 중 모바일에서 좋아요를 받은 비율을 구하고 그 비율이 0.6보다 크고 0.7보다 작으면서 성별이 남자인 경우의 레코드 수를 구하여라. (여기서, mobile_like_recived와 www_like_recived는 각각 모바일과 웹에서 좋아요를 받은 횟수이고 두 컬럼의 합은 좋아요를 받은 전체 수임)

# In[18]:


#### 기출복원 04회 02 Solution
import pandas as pd
exam2 = pd.read_csv('data/기출복원/04회/facebook_subset.csv')

##### 좋아요를 받은 전체 수 중 모바일에서 좋아요를 받은 비율
ratio = exam2['mobile_likes_received']/(exam2['mobile_likes_received'] + exam2['www_likes_received'])

##### 해당 비율이 0.6보다 크고 0.7보다 작은 경우
case1 = (ratio > 0.6) & (ratio < 0.7)

##### 성별이 남자인 경우
case2 = exam2['gender'] == 'male'

##### 결과를 result에 할당 
result = exam2[case1 & case2].shape[0]

##### 결과 출력
result


# --- 

# #### 3. netflix_subset 데이터셋을 불러와 2021년 1월에 등록되었으면서 listed_in이 오직 Drama인 경우의 레코드 수를 구하여라. 

# In[19]:


#### 기출복원 04회 03 Solution
import pandas as pd
exam3 = pd.read_csv('data/기출복원/04회/netflix_subset.csv', encoding = 'cp949')

##### 2021년 1월에 등록된 경우
# date_added : 등록일, release_year : 개봉일

# object -> datetime형으로 변환
exam3['date_added'] = pd.to_datetime(exam3['date_added'])

# 연/월 추출
year = exam3['date_added'].dt.year
month = exam3['date_added'].dt.month

case1 = (year == 2021) & (month == 1) 

##### listed_in이 오직 Drama인 경우
case2 = (exam3['listed_in'] == 'Dramas')

##### 결과를 result에 할당 
result = exam3[case1 & case2].shape[0]

##### 결과 출력
result


# In[20]:


#### 기출복원 04회 03 Solution
import pandas as pd
exam3 = pd.read_csv('data/기출복원/04회/netflix_subset.csv', encoding = 'cp949')

##### 2021년 1월에 등록된 경우
# date_added에 ' 22-Jan-21'과 ' 20-Jan-21'과 같이 공백이 함께 섞여있는 케이스들이 있기 때문에
# datetime형으로 변경하지 않고 문자열로 할 경우 공백을 처리하는 등의 방법을 함께 고려해야 함을 명심

# 선행 문자 (공백) 제거
exam3['date_added'] = exam3['date_added'].str.lstrip()

# 연/월 추출
# 비교적 일정한 형태의 형식이므로 간단히 인덱싱으로 해결 가능함
month = exam3['date_added'].str[3:6]
year = exam3['date_added'].str[-2:]

case1 = (year == '21') & (month == 'Jan') 

##### listed_in이 오직 Drama인 경우
case2 = (exam3['listed_in'] == 'Dramas')

##### 결과를 result에 할당 
result = exam3[case1 & case2].shape[0]

##### 결과 출력
result


# ---

# ### (작업형2) 

# ### 1. 아래는 연령별 수행등급을 확인한 자료와 일부 운동 수행도 데이터이다. 주어진 데이터를 이용하여 예측 모형을 만들고 아래에 따라 CSV 파일을 생성하시오. (단, 제출 전 두 개 이상의 모형의 성능을 비교하여 가장 우수한 모형을 선정할 것)

# In[21]:


#### 기출복원 04회차_작업형2 Solution

##### STEP1. 데이터셋 불러오기
import pandas as pd
train = pd.read_csv('data/기출복원/04회/bodyPerfor_train.csv')
test = pd.read_csv('data/기출복원/04회/bodyPerfor_test.csv')


# In[22]:


# train데이터를 X_train과 y_train으로 분할
X_train = train.drop(columns = 'class').copy()
y_train = train['class'].copy()

# test도 통일을 위해 X_test로 할당
X_test = test.copy()


# In[23]:


##### STEP2. 데이터셋 확인하기
###### STEP2-1. 데이터셋 일부 확인
print(X_train.head())
print(X_test.head())
print(y_train.head())


# In[24]:


###### STEP2-2. 데이터셋 요약 정보 확인
print(X_train.info())
print(X_test.info())
print(y_train.info())


# In[25]:


###### STEP2-3. 기초통계량 확인
print(X_train.describe())
print(X_test.describe())
print(y_train.describe())


# In[26]:


##### STEP3. 데이터셋 전처리
###### STEP3-1. 불필요한 컬럼 삭제
# id 컬럼은 개인의 고유번호로 key 역할로 모델에는 불필요함

# 데이터들에서 id 컬럼 삭제
X_train = X_train.drop(columns = 'id')
X_test = X_test.drop(columns = 'id')
y_train = y_train.drop(columns = 'id')


# In[27]:


###### STEP3-2. 결측치 처리
# 결측치 확인
X_train.isna().sum()


# In[28]:


X_test.isna().sum()


# In[29]:


# 결측치가 존재하지 않음


# In[30]:


###### STEP3-3. 카테고리형 컬럼 전처리
# 별도 과정이 없으므로 생략


# In[31]:


###### STEP3-4. 수치형 컬럼 전처리
####### age 컬럼
# 비닝하여 파생변수 age_gp에 할당, object형으로 변경
X_train['age_gp'] = pd.cut(X_train['age'], bins = list(range(0,70,10))).astype('object')
X_test['age_gp'] = pd.cut(X_test['age'], bins = list(range(0,70,10))).astype('object')

# 완료 후 삭제
X_train = X_train.drop('age', axis = 1)
X_test = X_test.drop('age', axis = 1)

####### height, weight, body_fat, diastolic, systolic, grip_force, sit_bend_forward, sit_ups, broad_jump
# 수치형 컬럼만 추출
X_train_conti = X_train.select_dtypes(exclude = 'object')
X_test_conti = X_test.select_dtypes(exclude = 'object')

# 상관관계 확인 컬럼간 강한 상관관계는 나타나지 않음
# X_train_conti.corr()


# In[32]:


###### STEP3-5. 데이터 분할
from sklearn.model_selection import train_test_split

# X_train과 y_train을 학습용(X_TRAIN, y_TRAIN)과 검증용(X_VAL, y_VAL)로 분할
X_TRAIN, X_VAL, y_TRAIN, y_VAL = train_test_split(X_train, y_train, random_state = 1234,
                                                 test_size = 0.3,
                                                 stratify = y_train)

# 분할 후 shape 확인
print(X_TRAIN.shape)
print(X_VAL.shape)
print(y_TRAIN.shape)
print(y_VAL.shape)


# In[33]:


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


# In[34]:


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


# In[35]:


###### STEP3-8. 입력 데이터셋 준비
import numpy as np

# 인코딩과 스케일링된 넘파이 배열 연결
X_TRAIN = np.concatenate([X_TRAIN_OH, X_TRAIN_STD], axis = 1)
X_VAL = np.concatenate([X_VAL_OH, X_VAL_STD], axis = 1)

# 'A'~'D'를 0~3로 매핑
y_TRAIN = y_TRAIN.map({'A':0, 'B':1, 'C':2, 'D':3})
y_VAL = y_VAL.map({'A':0, 'B':1, 'C':2, 'D':3})

# 1차원 넘파이 배열로 평탄화
y_TRAIN = y_TRAIN.values.ravel()
y_VAL = y_VAL.values.ravel()


# In[36]:


##### STEP4. 모델 학습
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

###### STEP4-1. random forest
rf = RandomForestClassifier(n_estimators = 500,
                            max_depth = 3,
                            min_samples_leaf = 10,
                            max_features = 'sqrt',
                            random_state = 2022)

model_rf = rf.fit(X_TRAIN, y_TRAIN)

###### STEP4-2. XGBoost
xgb = XGBClassifier(max_depth = 8,
                    n_estimators = 500,
                    nthread = 5,
                    min_child_weight = 20,
                    gamma = 0.5,
                    objective = 'multi:softmax',
                    use_label_encoder = False,
                    random_state = 2022)

model_xgb = xgb.fit(X_TRAIN, y_TRAIN, eval_metric = 'mlogloss')

###### STEP4-3. LightGBM
lgb = LGBMClassifier(max_depth = 8,
                     n_estimators = 500,
                     n_jobs = 30,
                     min_child_weight = 10,
                     learning_rate = 0.2,
                     objective = 'multiclass',
                     random_state = 2022)

model_lgb = lgb.fit(X_TRAIN, y_TRAIN)


# In[37]:


###### STEP4-4. 성능평가(기준:macro f1-score)를 통한 모델 선정
from sklearn.metrics import f1_score

# 검증용 데이터셋을 통한 예측
pred_rf = model_rf.predict(X_VAL)
pred_xgb = model_xgb.predict(X_VAL)
pred_lgb = model_lgb.predict(X_VAL)

# macro f1-score 계산
f1_rf = f1_score(y_VAL, pred_rf, average = 'macro')
print(f1_rf)

f1_xgb = f1_score(y_VAL, pred_xgb, average = 'macro')
print(f1_xgb)

f1_lgb = f1_score(y_VAL, pred_lgb, average = 'macro')
print(f1_lgb)


# In[38]:


##### STEP5. 결과 제출하기
###### 실제 시험에서 답 제출시에는 성능이 가장 우수한 모형 하나만 구현!
X_TEST = np.concatenate([X_TEST_OH, X_TEST_STD], axis = 1)
y_pred = model_xgb.predict(X_TEST)

# 문제에서 요구하는 형태로 변환 필요
obj = {'class' : y_pred}
result = pd.DataFrame(obj)
result['class'] = result['class'].map({0:'A', 1:'B', 2:'C', 3:'D'})

# 하위에 12345.csv 이름으로 저장하기
result.to_csv("12345.csv", index = False)


# In[39]:


##### STEP6. 채점 모델 평가(번외)
# 실제값
actual = pd.read_csv('data/기출복원/04회/bodyPerfor_class.csv')
actual = actual['class'].ravel()

# 채점 기준이 될 성과지표 값
f1_score(actual, result['class'], average = 'macro')


# # (끝)
