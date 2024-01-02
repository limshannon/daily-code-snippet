#!/usr/bin/env python
# coding: utf-8

# ## 모의고사 3회

# ### (작업형1) 

# #### 1. Rabbit 데이터셋을 불러와 Dose 컬럼의 제 3사분위수와 제 2사분위수를 구하고 두 값의 차이의 절댓값을 구한 후 소수점을 버린 값을 출력하여라.

# In[1]:


#### 모의고사 03회 01 Solution
import pandas as pd
exam1 = pd.read_csv('data/모의고사/03회/Rabbit.csv')

##### 제 3사분위수, 제 2사분위수 별도 저장
q3 = exam1['Dose'].quantile(.75)
q2 = exam1['Dose'].median()

##### 두 값의 차이의 절댓값
diff = abs(q3 - q2)

##### 결과를 result에 할당
result = diff.astype('int64')

##### 결과 출력
print(result)


# --- 

# #### 2. Boston 데이터셋을 불러와 medv 컬럼에 대해서 동일한 폭으로 binning한 후 가장 많은 빈도를 가지는 구간을 산출하고 해당 구간 내 dis 컬럼의 중앙값을 구하여라. (폭은 10을 기준으로 하고 소수점은 둘째 자리까지 나타내시오)

# In[2]:


#### 모의고사 03회 02 Solution
import pandas as pd
exam3 = pd.read_csv('data/모의고사/03회/Boston.csv')

##### medv 컬럼에 대해서 동일한 폭으로 binning
medv_cut = pd.cut(exam3['medv'], bins = [0, 10, 20, 30, 40, 50])

#### 가장 많은 빈도를 가지는 구간을 산출
mode = medv_cut.value_counts().idxmax()

#### 해당구간 내 dis 컬럼의 중앙값
# 조건 
cond = (medv_cut == mode)

# 중앙값
median = exam3['dis'][cond].median()

##### 결과를 result에 할당
result = round(median, 2)

##### 결과 출력
print(result)


# --- 

# #### 3. Melanoma 데이터셋을 불러와 1번째-123번째 레코드와 123번째 이후 레코드로 데이터셋을 분리하고 각 데이터셋별로 thickness 컬럼을 z-score 정규화로 변환한 후 –1과 1 사이 값들의 중앙값을 각각 산출한 후 합계를 구하여라.(단, z-score 정규화 변환 계산에 사용되는 평균과 표준편차는 분리된 것과 관계 없이 1번째~123번째 레코드로 이루어진 데이터셋을 기준으로 하고 출력시 소수점 넷째 자리까지 반올림하여 나타낼 것, 레코드 번호는 가장 위에 위치한 레코드를 1번으로 가정함)

# In[3]:


#### 모의고사 03회차 03 Solution
import pandas as pd
exam3 = pd.read_csv('data/모의고사/03회/Melanoma.csv')

##### 1번째 ~ 123번째 레코드와 123번째 이후 레코드로 데이터셋을 분리
df1 = exam3.iloc[:123]
df2 = exam3.iloc[123:]

##### thickness 컬럼을 z-score 정규화로 변환
# 1번째~123번째 레코드로 이루어진 데이터셋의 thickness 평균
avg = df1['thickness'].mean()

# 1번째~123번째 레코드로 이루어진 데이터셋의 thickness 표준편차
sd = df1['thickness'].std()

# z-score 변환
std1 = (df1['thickness'] - avg)/sd
std2 = (df2['thickness'] - avg)/sd

##### –1과 1 사이 값들의 중앙값을 각각 산출
# -1과 1사이 값
sub_std1 = std1[(std1 > -1) & (std1 < 1)]
sub_std2 = std2[(std2 > -1) & (std2 < 1)]

# 중앙값
med1 = sub_std1.median()
med2 = sub_std2.median()

##### 결과를 result에 할당
result = round(med1 + med2, 4)

##### 결과 출력
print(result)


# In[22]:


# 데이터 분리된 것 확인하는 코드, 실제 시험에서는 실행 안 되게 해야함
print(df1), print(df2)


# ---

# ### (작업형2) 

# ### 1. 아래는 호주의 기상 관측소들의 일자별 기상 정보와 강수 여부에 관련한 데이터의 일부이다. 주어진 데이터를 이용하여 예측 모형을 만들고 아래에 따라 CSV 파일을 생성하시오.

# In[5]:


#### 모의고사 03회차_작업형2 Solution

##### STEP1. 데이터셋 불러오기
import pandas as pd
X_train = pd.read_csv('data/모의고사/03회/weatherAUS_X_train.csv')
X_test = pd.read_csv('data/모의고사/03회/weatherAUS_X_test.csv')
y_train = pd.read_csv('data/모의고사/03회/weatherAUS_y_train.csv')


# In[6]:


##### STEP2. 데이터셋 확인하기
###### STEP2-1. 데이터셋 일부 확인
print(X_train.head())
print(X_test.head())
print(y_train.head())


# In[7]:


###### STEP2-2. 데이터셋 요약 정보 확인
print(X_train.info())
print(X_test.info())
print(y_train.info())


# In[8]:


###### STEP2-3. 기초통계량 확인
print(X_train.describe())
print(X_test.describe())
print(y_train.describe())


# In[9]:


##### STEP3. 데이터셋 전처리
###### STEP3-1. 불필요한 컬럼 삭제
# Date 컬럼은 관측일자로 key 역할로 모델에는 불필요함
# 결과 제출 시에는 X_test의 Date 컬럼이 필요하기 때문에 별도 저장
Date = X_test['Date'].copy()

# 데이터들에서 Date 컬럼 삭제
X_train = X_train.drop(columns = 'Date')
X_test = X_test.drop(columns = 'Date')
y_train = y_train.drop(columns = 'Date')


# In[10]:


###### STEP3-2. 결측치 처리
# 결측치 확인
X_train.isna().sum()


# In[11]:


X_test.isna().sum()


# In[12]:


####### train에서 500개가 넘는 결측치가 있는 컬럼은 삭제
# 결측치가 500개가 넘는 조건
cond_na500 = (X_train.isna().sum() >= 500)

# 500개가 넘는 컬럼명
col_na500 = X_train.columns[cond_na500]

# 컬럼 삭제
X_train = X_train.drop(col_na500, axis = 1)
X_test = X_test.drop(col_na500, axis = 1)


# In[13]:


###### train에서 100개 미만의 결측치가 있는 컬럼은 결측치 대체
# 결측치가 100개 미만인 조건
X_train.isna().sum() < 100


# In[14]:


######## 수치형인 MinTemp, MaxTemp, Rainfall, WindSpeed9am, Humidity9am, Temp9am은 평균 대체
# 수치형만 있는 데이터프레임 추출
X_train_conti = X_train.select_dtypes(exclude = 'object').copy()
X_test_conti = X_test.select_dtypes(exclude = 'object').copy()

# 평균대치
X_train_conti = X_train_conti.fillna(X_train_conti.mean())
X_test_conti = X_test_conti.fillna(X_train_conti.mean())

######## 카테고리형인 RainToday는 최다빈도를 가지는 라벨로 대체
# 카테고리형만 있는 데이터프레임 추출
X_train_category = X_train.select_dtypes('object').copy()
X_test_category = X_test.select_dtypes('object').copy()

# 최다라벨로 대치
mode = X_train_category.value_counts('RainToday').idxmax()

X_train_category = X_train_category.fillna(mode)
X_test_category = X_test_category.fillna(mode)

######## 두 데이터 프레임 다시 합치기
X_train = pd.concat([X_train_conti, X_train_category], axis = 1)
X_test = pd.concat([X_test_conti, X_test_category], axis = 1)


# In[15]:


###### STEP3-3. 카테고리형 컬럼 전처리
# 문자열(object) 컬럼들의 유일값 수 확인
# .select_dtypes()은 원하는 dtype을 가진 데이터프레임만 추출 후 유일값 수 확인
# 별도의 과정 없음


# In[16]:


###### STEP3-4. 수치형 컬럼 전처리
# 별도의 과정 없음


# In[17]:


###### STEP3-5. 데이터 분할
from sklearn.model_selection import train_test_split

# X_train과 y_train을 학습용(X_TRAIN, y_TRAIN)과 검증용(X_VAL, y_VAL)로 분할
X_TRAIN, X_VAL, y_TRAIN, y_VAL = train_test_split(X_train, y_train, random_state = 1234, test_size = 0.3)

# 분할 후 shape 확인
print(X_TRAIN.shape)
print(X_VAL.shape)
print(y_TRAIN.shape)
print(y_VAL.shape)


# In[18]:


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


# In[19]:


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


# In[20]:


###### STEP3-8. 입력 데이터셋 준비
import numpy as np

# 인코딩과 스케일링된 넘파이배열 연결
X_TRAIN = np.concatenate([X_TRAIN_OH, X_TRAIN_STD], axis = 1)
X_VAL = np.concatenate([X_VAL_OH, X_VAL_STD], axis = 1)

# 'Yes'와 'No'를 각각 1,0에 매핑
y_TRAIN = y_TRAIN['RainTomorrow'].map({'No':0, 'Yes':1})
y_VAL = y_VAL['RainTomorrow'].map({'No':0, 'Yes':1})

# 1차원 넘파이 배열로 평탄화
y_TRAIN = y_TRAIN.values.ravel()
y_VAL = y_VAL.values.ravel()


# In[21]:


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


# In[ ]:


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


# In[ ]:


##### STEP5. 결과 제출하기
###### 실제 시험에서 답 제출시에는 성능이 가장 우수한 모형 하나만 구현!
X_TEST = np.concatenate([X_TEST_OH, X_TEST_STD], axis = 1)
y_score = model_lgb.predict_proba(X_TEST)[:,1]

# 문제에서 요구하는 형태로 변환 필요
obj = {'Date' : Date,
       'RainTomorrow_prob' : y_score}

result = pd.DataFrame(obj)

# 하위에 12345.csv 이름으로 저장하기
result.to_csv("12345.csv", index = False)


# In[ ]:


##### STEP6. 채점 모델 평가(번외)
# 실제값
actual = pd.read_csv('data/모의고사/03회/weatherAUS_y_test.csv', encoding = 'cp949')
actual = actual['RainTomorrow'].ravel()

# 채점 기준이 될 성과지표 값
fpr, tpr, thresholds = roc_curve(actual, y_score, pos_label = 'Yes')
auc(fpr, tpr)


# # (끝)
