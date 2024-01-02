#!/usr/bin/env python
# coding: utf-8

# ## 기출복원 2회

# ### (작업형1) 

# #### 1. mtcars2 데이터셋을 불러와 mpg 컬럼의 상위 10번째 값으로 상위 10개 값을 변환한 후 drat가 4 이상인 값에 대해 mpg의 평균을 구하여라. (단, 소수점은 반올림하여 셋째 자리까지 나타낼 것)

# In[4]:


#### 기출복원 02회 01 Solution
import pandas as pd
exam1 = pd.read_csv('data/기출복원/02회/mtcars2.csv')

##### mpg와 drat 컬럼 별도 할당
mpg = exam1['mpg'].copy()
drat = exam1['drat'].copy()

##### mpg 컬럼의 상위 10번째 값
tenth = mpg.sort_values(ascending = False, ignore_index = True)[9]

##### 상위 10번째 값으로 상위 10개 값을 변환
# 상위 10개의 인덱스
idx = mpg.sort_values(ascending = False)[:10].index

# 상위 10개 값 변경
mpg[idx] = tenth

##### drat가 4이상일 때 mpg의 평균
avg_mpg = mpg[drat >= 4].mean()

##### 결과 저장
result = round(avg_mpg, 3)

##### 결과 출력
print(result)


# --- 

# #### 2. 앞의 데이터셋을 새로 불러와 첫 번째 행부터 순서대로 80%까지의 데이터를 훈련 데이터로 추출한 후 disp 컬럼의 결측값을 disp 컬럼의 중앙값으로 대체하고 대체 전 후의 disp 변수의 표준편차 값의 차이를 구하여라. (단, 차이는 빼는 순서와 관계없이 절댓값을 취하여 표시하고 소수점은 넷째 자리에서 반올림할 것)

# In[5]:


#### 기출복원 02회 02 Solution
import pandas as pd
import numpy as np
exam2 = pd.read_csv('data/기출복원/02회/mtcars2.csv')

#### 첫 번째 행부터 순서대로 80%까지의 데이터를 훈련 데이터로 추출
# 80%가 되는 행 인덱스 번호 찾기
idx80 = np.floor(exam2.shape[0] * 0.8).astype('int')

# 훈련 데이터 추출
train = exam2[:idx80]

#### disp 변수의 결측값 중앙값으로 대체, 대체 전후 표준편차
# disp 컬럼 할당
disp_before = train['disp'].copy() # 대체하지 않을 것
disp_after = train['disp'].copy() # 대체할 것

# 대체 전(disp_before)의 중앙값과 표준편차
med_before = disp_before.median()
sd_before = disp_before.std()

# 결측치 중앙값 대체
disp_after = disp_after.fillna(med_before)

# 대체 후(disp_after)의 표준편차
sd_after = disp_after.std()

# 차이
diff = abs(sd_before - sd_after)

##### 결과 저장
result = round(diff, 3)

##### 결과 출력
print(result)


# --- 

# #### 3. gehan 데이터 셋을 불러와 time 컬럼에서 이상값의 합을 구하여라.(단, 이상값은 평균에서 1.5 표준편차 이상으로 벗어난 값으로 정의함)

# In[6]:


#### 기출복원 02회 03 Solution
import pandas as pd
exam3 = pd.read_csv('data/기출복원/02회/gehan.csv')

##### time 컬럼의 평균과 표준편차
time = exam3['time']
avg_time = time.mean()
sd_time = time.std()

##### 평균에서 1.5 표준편차 구간
low = avg_time - 1.5 * sd_time
upp = avg_time + 1.5 * sd_time

##### 이상치
outlier = time[(time < low) | (time > upp)]

##### 결과를 result에 할당
result = sum(outlier)

##### 결과 출력
print(result)


# ---

# ### (작업형2) 

# ### 1. 아래는 뇌졸중(stroke)에 대한 환자들의 임상적 변수에 관련한 데이터의 일부이다. 주어진 데이터를 이용하여 예측 모형을 만들고 아래에 따라 CSV 파일을 생성하시오. (단, 제출 전 두 개이상의 모형의 성능을 비교하여 가장 우수한 모형을 선정할 것)

# In[1]:


#### 기출복원 02회차 작업형2 Solution

##### STEP1. 데이터셋 불러오기
import pandas as pd
X_train = pd.read_csv('data/기출복원/02회/stroke_X_train.csv')
X_test = pd.read_csv('data/기출복원/02회/stroke_X_test.csv')
y_train = pd.read_csv('data/기출복원/02회/stroke_y_train.csv')


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


# In[4]:


###### STEP2-3. 기초통계량 확인
print(X_train.describe())
print(X_test.describe())
print(y_train.describe())


# In[5]:


##### STEP3. 데이터셋 전처리
###### STEP3-1. 불필요한 컬럼 삭제
# id 컬럼은 환자에 대한 고유 정보로 key 역할로 모델에는 불필요함
# 결과 제출 시에는 X_test의 id 컬럼이 필요하기 때문에 별도 저장
ID = X_test['id'].copy()

# 데이터들에서 id 컬럼 삭제
X_train = X_train.drop(columns = 'id')
X_test = X_test.drop(columns = 'id')
y_train = y_train.drop(columns = 'id')


# In[6]:


###### STEP3-2. 결측치 처리
# 결측치 확인
X_train.isna().sum()


# In[7]:


X_test.isna().sum()


# In[8]:


# smoking_status 내 “Unknown”은 정보에 대해 알 수 없는 것으로 결측임
X_train['smoking_status'].value_counts()


# In[9]:


X_test['smoking_status'].value_counts()


# In[10]:


####### bmi 컬럼 (train:165, test:36 결측)
# 평균값 대치
avg_bmi = X_train['bmi'].mean()

X_train['bmi'] = X_train['bmi'].fillna(avg_bmi)
X_test['bmi'] = X_test['bmi'].fillna(avg_bmi)

###### smoking_status 컬럼 (train:1,240, test:304 결측)
# 컬럼 삭제
X_train = X_train.drop('smoking_status', axis = 1)
X_test = X_test.drop('smoking_status', axis = 1)


# In[11]:


###### STEP3-3. 카테고리형 컬럼 전처리
#별도 과정이 없으므로 생략


# In[12]:


###### STEP3-4. 수치형 컬럼 전처리
####### age 컬럼
# 비닝하여 파생변수 age_gp에 할당, object형으로 변경
X_train['age_gp'] = pd.cut(X_train['age'], bins = list(range(0,100,10))).astype('object')
X_test['age_gp'] = pd.cut(X_test['age'], bins = list(range(0,100,10))).astype('object')

# 완료 후 삭제
X_train = X_train.drop('age', axis = 1)
X_test = X_test.drop('age', axis = 1)


# In[13]:


###### STEP3-5. 데이터 분할
from sklearn.model_selection import train_test_split

# X_train과 y_train을 학습용(X_TRAIN, y_TRAIN)과 검증용(X_VAL, y_VAL)로 분할
X_TRAIN, X_VAL, y_TRAIN, y_VAL = train_test_split(X_train, y_train, random_state = 1234, test_size = 0.2)

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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

###### STEP4-1. random forest
rf = RandomForestClassifier(n_estimators = 500,
                            max_depth = 3,
                            min_samples_leaf = 10,
                            max_features = 2,
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


# In[18]:


###### STEP4-4. 성능평가(기준:accuracy_score)를 통한 모델 선정
from sklearn.metrics import accuracy_score

# 검증용 데이터셋을 통한 예측
pred_rf = model_rf.predict(X_VAL)
pred_xgb = model_xgb.predict(X_VAL)
pred_lgb = model_lgb.predict(X_VAL)

# accuracy 계산
acc_rf = accuracy_score(y_VAL, pred_rf)
print(acc_rf)

acc_xgb = accuracy_score(y_VAL, pred_xgb)
print(acc_xgb)

acc_lgb = accuracy_score(y_VAL, pred_lgb)
print(acc_lgb)


# In[19]:


##### STEP5. 결과 제출하기
###### 실제 시험에서 답 제출시에는 성능이 가장 우수한 모형 하나만 구현!
X_TEST = np.concatenate([X_TEST_OH, X_TEST_STD], axis = 1)
y_pred = model_lgb.predict(X_TEST)

# 문제에서 요구하는 형태로 변환 필요
obj = {'id' : ID,
       'stroke' : y_pred}
result = pd.DataFrame(obj)

# 하위에 12345.csv 이름으로 저장하기
result.to_csv("12345.csv", index = False)


# In[20]:


##### STEP6. 채점 모델 평가(번외)
# 실제값
actual = pd.read_csv('data/기출복원/02회/stroke_y_test.csv')
actual = actual['stroke'].ravel()

# 채점 기준이 될 성과지표 값
accuracy_score(actual, y_pred)


# # (끝)
