#!/usr/bin/env python
# coding: utf-8

# ## 기출복원 3회

# ### (작업형1) 

# #### 1. economics 데이터셋을 불러와 첫 번째 행부터 순서대로 70%까지의 데이터를 훈련 데이터로 추출한 후 pce 컬럼의 제 1사분위수를 구하라. (단, 정수로 나타낼 것)

# In[1]:


#### 기출복원 03회 01 Solution
import pandas as pd
import numpy as np

exam1 = pd.read_csv('data/기출복원/03회/economics.csv')

#### 첫 번째 행부터 순서대로 70%까지의 데이터를 훈련 데이터로 추출
# 70%가 되는 행 인덱스 번호 찾기
idx70 = np.floor(exam1.shape[0] * 0.7).astype('int')

# 훈련 데이터 추출
train = exam1[:idx70]

##### pce 컬럼의 제 1사분위수
q1 = train['pce'].quantile(.25)

##### 결과를 result에 할당
result = q1.astype('int')

##### 결과 출력
print(result)


# --- 

# #### 2. Hitters 데이터셋을 불러와 Years 컬럼이 10인 데이터만 추출하여 HmRun 컬럼이 평균보다 큰 선수가 몇 명인지 계산하라.

# In[2]:


#### 기출복원 03회 02 Solution
import pandas as pd

exam2 = pd.read_csv('data/기출복원/03회/Hitters.csv')

#### Years 컬럼이 10인 데이터만 추출
subset = exam2[exam2['Years'] == 10]

#### subset에서 HmRun 컬럼이 HmRun 평균보다 큰 경우
avg_HmRun = subset['HmRun'].mean()
case = (subset['HmRun'] > avg_HmRun)

##### 결과를 result에 할당
result = subset[case].shape[0]

##### 결과 출력
print(result)


# #### 3. msleep 데이터셋을 불러와 가장 결측치가 많은 컬럼의 이름을 출력하라.

# In[3]:


#### 기출복원 03회 03 Solution
import pandas as pd

exam3 = pd.read_csv('data/기출복원/03회/msleep.csv')

#### 컬럼명 결측치
col_na = exam3.isna().sum()

##### 결과를 result에 할당
# 가장 결측치가 많은 컬럼
result = col_na.idxmax()

##### 결과 출력
print(result)


# ---

# ### (작업형2) 

# ### 1. 아래는 HR 연구를 위한 이직 희망 여부와 입사 지원자들의 정보와 관련한 데이터의 일부이다.주어진 데이터를 이용하여 예측 모형을 만들고 아래에 따라 CSV 파일을 생성하시오.

# In[4]:


#### 기출복원 03회차_작업형2 Solution

##### STEP1. 데이터셋 불러오기
import pandas as pd
X_train = pd.read_csv('data/기출복원/03회/job_change_X_train.csv')
X_test = pd.read_csv('data/기출복원/03회/job_change_X_test.csv')
y_train = pd.read_csv('data/기출복원/03회/job_change_y_train.csv')


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
# enrollee_id 컬럼은 지원자의 고유 정보로 key 역할로 모델에는 불필요함
# 결과 제출 시에는 X_test의 enrollee_id 컬럼이 필요하기 때문에 별도 저장
enrollee_id = X_test['enrollee_id'].copy()

# 데이터들에서 enrollee_id 컬럼 삭제
X_train = X_train.drop(columns = 'enrollee_id')
X_test = X_test.drop(columns = 'enrollee_id')
y_train = y_train.drop(columns = 'enrollee_id')


# In[9]:


###### STEP3-2. 결측치 처리
# 결측치 확인
X_train.isna().sum()


# In[10]:


X_test.isna().sum()


# In[11]:


####### train에서 1000개가 넘는 공백이 있는 컬럼은 삭제
# 1000개가 넘을 조건
cond_na1000 = (X_train.isna().sum() > 1000)

# 1000개가 넘는 컬럼명
colnm_na1000 = X_train.columns[cond_na1000]

# 컬럼 삭제
X_train = X_train.drop(colnm_na1000, axis = 1)
X_test = X_test.drop(colnm_na1000, axis = 1)


# In[12]:


####### train에서 200개 미만의 결측치가 있는 컬럼은 결측치 대체
######## enrolled_university 컬럼 (train : 142, test : 51)
# 최다빈도를 가지는 라벨로 대체
mode_EU = X_train['enrolled_university'].value_counts().idxmax()
X_train['enrolled_university'] = X_train['enrolled_university'].fillna(mode_EU)
X_test['enrolled_university'] = X_test['enrolled_university'].fillna(mode_EU)

######## education_level 컬럼 (train : 162, test : 82)
# 최다빈도를 가지는 라벨로 대체
mode_EL = X_train['education_level'].value_counts().idxmax()
X_train['education_level'] = X_train['education_level'].fillna(mode_EU)
X_test['education_level'] = X_test['education_level'].fillna(mode_EU)


# In[13]:


###### STEP3-3. 카테고리형 컬럼 전처리
# 별도 과정이 없으므로 생략


# In[14]:


###### STEP3-4. 수치형 컬럼 전처리
# 전처리할 수치형 컬럼 없으므로 생략


# In[15]:


###### STEP3-5. 데이터 분할
from sklearn.model_selection import train_test_split

# X_train과 y_train을 학습용(X_TRAIN, y_TRAIN)과 검증용(X_VAL, y_VAL)로 분할
X_TRAIN, X_VAL, y_TRAIN, y_VAL = train_test_split(X_train, y_train, random_state = 1234, test_size = 0.2)

# 분할 후 shape 확인
print(X_TRAIN.shape)
print(X_VAL.shape)
print(y_TRAIN.shape)
print(y_VAL.shape)


# In[16]:


###### STEP3-6. 인코딩
# 카테고리형 컬럼에 대하여 원-핫 인코딩 수행
from sklearn.preprocessing import OneHotEncoder

# 인코딩할 카테고리형 컬럼만 별도 저장
X_TRAIN_category = X_TRAIN.select_dtypes('object').copy()
X_VAL_category = X_VAL.select_dtypes('object').copy()
X_TEST_category = X_test.select_dtypes('object').copy()

# 원-핫 인코딩
enc = OneHotEncoder(sparse = False, handle_unknown = 'ignore').fit(X_TRAIN_category)

X_TRAIN_OH = enc.transform(X_TRAIN_category)
X_VAL_OH = enc.transform(X_VAL_category)
X_TEST_OH = enc.transform(X_TEST_category)


# In[17]:


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


# In[18]:


###### STEP3-8. 입력 데이터셋 준비
import numpy as np

# 인코딩과 스케일링된 넘파이 배열 연결
X_TRAIN = np.concatenate([X_TRAIN_OH, X_TRAIN_STD], axis = 1)
X_VAL = np.concatenate([X_VAL_OH, X_VAL_STD], axis = 1)

# 1차원 넘파이 배열로 평탄화
y_TRAIN = y_TRAIN.values.ravel()
y_VAL = y_VAL.values.ravel()


# In[19]:


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
                    objective = 'binary:logistic',
                    use_label_encoder = False,
                    random_state = 2022)

model_xgb = xgb.fit(X_TRAIN, y_TRAIN, eval_metric = 'mlogloss')

###### STEP4-3. LightGBM
lgb = LGBMClassifier(max_depth = 8,
                     n_estimators = 500,
                     n_jobs = 30,
                     min_child_weight = 10,
                     learning_rate = 0.2,
                     objective = 'binary',
                     random_state = 2022)

model_lgb = lgb.fit(X_TRAIN, y_TRAIN)


# In[20]:


###### STEP4-4. 성능평가(기준:AUC)를 통한 모델 선정
from sklearn.metrics import roc_curve, auc

# 검증용 데이터셋을 통한 예측
score_rf = model_rf.predict_proba(X_VAL)[:,1]
score_xgb = model_xgb.predict_proba(X_VAL)[:,1]
score_lgb = model_lgb.predict_proba(X_VAL)[:,1]

# AUC 계산
fpr, tpr, thresholds = roc_curve(y_VAL, score_rf)
auc_rf = auc(fpr, tpr)
print(auc_rf)

fpr, tpr, thresholds = roc_curve(y_VAL, score_xgb)
auc_xgb = auc(fpr, tpr)
print(auc_xgb)

fpr, tpr, thresholds = roc_curve(y_VAL, score_lgb)
auc_lgb = auc(fpr, tpr)
print(auc_lgb)


# In[21]:


##### STEP5. 결과 제출하기
###### 실제 시험에서 답 제출시에는 성능이 가장 우수한 모형 하나만 구현!
X_TEST = np.concatenate([X_TEST_OH, X_TEST_STD], axis = 1)
y_pred = model_rf.predict(X_TEST)
y_score = model_rf.predict_proba(X_TEST)[:,1]

# 문제에서 요구하는 형태로 변환 필요
obj = {'enrollee_id' : enrollee_id,
       'target' : y_pred, 
       'target_prob' : y_score}
result = pd.DataFrame(obj)

# 하위에 12345.csv 이름으로 저장하기
result.to_csv("12345.csv", index = False)


# In[22]:


##### STEP6. 채점 모델 평가(번외)
# 실제값
actual = pd.read_csv('data/기출복원/03회/job_change_y_test.csv')
actual = actual['target'].ravel()

# 채점 기준이 될 성과지표 값
fpr, tpr, thresholds = roc_curve(actual, y_score)
auc(fpr, tpr)


# # (끝)
