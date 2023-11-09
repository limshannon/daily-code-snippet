#!/usr/bin/env python
# coding: utf-8

# ## (작업형2) 연습문제

# #### 2. 아래는 1995년 US News and World Report에 따른 미국 대학별 사립 여부와 주요 속성에 관한 자료이다.

# In[3]:


#### 연습문제2 Solution

##### STEP1. 데이터셋 불러오기
import pandas as pd
X_train = pd.read_csv('data/연습문제/College_X_train.csv', encoding = 'cp949')
X_test = pd.read_csv('data/연습문제/College_X_test.csv', encoding = 'cp949')
y_train = pd.read_csv('data/연습문제/College_y_train.csv', encoding = 'cp949')


# In[4]:


##### STEP2. 데이터셋 확인하기
###### STEP2-1. 데이터셋 일부 확인
print(X_train.head())
print(X_test.head())
print(y_train.head())


# In[5]:


###### STEP2-2. 데이터셋 요약 정보 확인
print(X_train.info())
print(X_test.info())
print(y_train.info())


# In[6]:


###### STEP2-3. 기초통계량 확인
# 수치형 컬럼들의 기초통계 확인
print(X_train.describe())
print(X_test.describe())
print(y_train.describe())


# In[7]:


##### STEP3. 데이터셋 전처리
###### STEP3-1. 불필요한 컬럼 삭제
# ID 컬럼은 학교에 대한 고유 정보로 key 역할로 모델에는 불필요함
# Name 컬럼도 학교명에 대한 고유 정보로 key 역할로 모델에는 불필요함
# 결과 제출 시에는 X_test의 ID 컬럼이 필요하기 때문에 별도 저장
ID = X_test['ID'].copy()

# 데이터들에서 ID 컬럼 삭제
X_train = X_train.drop(columns = ['ID', 'Name'])
X_test = X_test.drop(columns = ['ID', 'Name'])
y_train = y_train.drop(columns = 'ID')


# In[8]:


###### STEP3-2. 결측치 처리
# 결측치 확인 및 공백확인
X_train.isna().sum()


# In[9]:


X_test.isna().sum()


# In[10]:


###### STEP3-4. 수치형 컬럼 전처리
# Top10perc, Top25perc, PhD, Terminal, S.F.Ratio, Grad.Rate, 'perc.alumni'
# 위 7개 컬럼은 비율에 관한 컬럼으로 단위는 백분율임
# 이를 소숫점으로 변환
col_per = ['Top10perc', 'Top25perc', 'PhD', 'Terminal', 'S.F.Ratio', 'Grad.Rate', 'perc.alumni']
X_train[col_per] = X_train[col_per]/100
X_test[col_per] = X_test[col_per]/100


# In[11]:


####### 수치형 컬럼 간 상관관계 확인
# 상관관계를 확인할 컬럼만
X_train.corr()


# In[12]:


# Apps, Accept, Enroll, F.Undergrad 컬럼 간
# Top10perc와 Top25perc, PhD와 Terminal 컬럼 간
# 상관관계 높음
# Apps, Accept, F.Undergrad  Top25perc, Terminal 컬럼 삭제
col_del = ['Apps', 'Accept', 'F.Undergrad','Top25perc', 'Terminal']
X_train = X_train.drop(col_del, axis = 1)
X_test = X_test.drop(col_del, axis = 1)


# In[13]:


###### STEP3-5. 데이터 분할
from sklearn.model_selection import train_test_split
# X_train과 y_train을 학습용(X_TRAIN, y_TRAIN)과 검증용(X_VAL, y_VAL)로 분할
X_TRAIN, X_VAL, y_TRAIN, y_VAL = train_test_split(X_train, y_train,
                                                  random_state = 1234,
                                                  test_size = 0.1,
                                                  stratify = y_train)

print(X_TRAIN.shape)
print(X_VAL.shape)
print(y_TRAIN.shape)
print(y_VAL.shape)


# In[14]:


###### STEP3-6. 인코딩
# 카테고리형 컬럼에 대하여 원-핫 인코딩 수행
# 없으므로 생략


# In[15]:


###### STEP3-7. 스케일링
from sklearn.preprocessing import StandardScaler

# TRAIN 데이터 기준으로 스케일링함
scale = StandardScaler().fit(X_TRAIN)

# z-점수 표준화
X_TRAIN_STD = scale.transform(X_TRAIN)
X_VAL_STD = scale.transform(X_VAL)
X_TEST_STD = scale.transform(X_test)


# In[16]:


###### STEP3-8. 입력 데이터셋 준비
X_TRAIN = X_TRAIN_STD
X_VAL = X_VAL_STD

y_TRAIN = y_TRAIN.values.ravel()
y_VAL = y_VAL.values.ravel()


# In[17]:


##### STEP4. 모델 학습
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

###### STEP4-1. random forest
rf = RandomForestClassifier(n_estimators = 500,
                           max_depth = 3,
                           min_samples_leaf = 10,
                           max_features = 'sqrt',
                           random_state = 2022)
model_rf = rf.fit(X_TRAIN, y_TRAIN)

###### STEP4-2. Bagging
dtr = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 10)
bag = BaggingClassifier(base_estimator = dtr,
                       n_estimators = 500,
                       random_state = 2022)
model_bag = bag.fit(X_TRAIN, y_TRAIN)

###### STEP4-3. AdaBoost
ada = AdaBoostClassifier(n_estimators = 500,
                        learning_rate = 0.5,
                        random_state = 2022)
model_ada = ada.fit(X_TRAIN, y_TRAIN)


# In[18]:


###### STEP4-4. 성능평가(기준:AUC)를 통한 모델 선정
from sklearn.metrics import roc_curve, auc

# 검증용 데이터셋을 통한 예측
score_rf = model_rf.predict_proba(X_VAL)[:,1]
score_bag = model_bag.predict_proba(X_VAL)[:,1]
score_ada = model_ada.predict_proba(X_VAL)[:,1]

# AUC 계산
fpr, tpr, thresholds = roc_curve(y_VAL, score_rf, pos_label = 'Yes')
auc_rf = auc(fpr, tpr)
print(auc_rf)

fpr, tpr, thresholds = roc_curve(y_VAL, score_bag, pos_label = 'Yes')
auc_bag = auc(fpr, tpr)
print(auc_bag)

fpr, tpr, thresholds = roc_curve(y_VAL, score_ada, pos_label = 'Yes')
auc_ada = auc(fpr, tpr)
print(auc_ada)


# In[19]:


##### STEP5. 결과 제출하기
###### 실제 시험에서 답 제출시에는 성능이 가장 우수한 모형 하나만 구현!
X_TEST = X_TEST_STD
y_score = model_ada.predict_proba(X_TEST)[:,1]

# 문제에서 요구하는 형태로 변환 필요
obj = {'ID' : ID,
       'prob_Private' : y_score}
result = pd.DataFrame(obj)

# 하위에 12345.csv 이름으로 저장하기
result.to_csv("12345.csv", index = False)


# In[20]:


##### STEP6. 채점 모델 평가(번외)
# 실제값
actual = pd.read_csv('data/연습문제/College_y_test.csv', encoding = 'cp949')
actual = actual['Private'].ravel()


# In[21]:


# 채점 기준이 될 성과지표 값
fpr, tpr, thresholds = roc_curve(actual, y_score, pos_label = 'Yes')
auc(fpr, tpr)


# ----

# #### 실제 시험에서 제출해야하는 답안

# In[22]:


#### 연습문제2 Solution

##### STEP1. 데이터셋 불러오기
import pandas as pd
X_train = pd.read_csv('data/연습문제/College_X_train.csv', encoding = 'cp949')
X_test = pd.read_csv('data/연습문제/College_X_test.csv', encoding = 'cp949')
y_train = pd.read_csv('data/연습문제/College_y_train.csv', encoding = 'cp949')

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
# ID 컬럼은 학교에 대한 고유 정보로 key 역할로 모델에는 불필요함
# Name 컬럼도 학교명에 대한 고유 정보로 key 역할로 모델에는 불필요함
# 결과 제출 시에는 X_test의 ID 컬럼이 필요하기 때문에 별도 저장
ID = X_test['ID'].copy()

# 데이터들에서 ID 컬럼 삭제
X_train = X_train.drop(columns = ['ID', 'Name'])
X_test = X_test.drop(columns = ['ID', 'Name'])
y_train = y_train.drop(columns = 'ID')

###### STEP3-2. 결측치 처리
# 결측치 확인 및 공백확인
# X_train.isna().sum()

###### STEP3-3. 카테고리형 컬럼 전처리
# 문자열(object) 컬럼들의 유일값 수 확인
# .select_dtypes()은 원하는 dtype을 가진 데이터프레임만 추출 후 유일값 수 확인
# print(X_train.select_dtypes('object').nunique())
# print(X_test.select_dtypes('object').nunique())

###### STEP3-4. 수치형 컬럼 전처리
# Top10perc, Top25perc, PhD, Terminal, S.F.Ratio, Grad.Rate, 'perc.alumni'
# 위 7개 컬럼은 비율에 관한 컬럼으로 단위는 백분율임
# 이를 소숫점으로 변환
col_per = ['Top10perc', 'Top25perc', 'PhD', 'Terminal', 'S.F.Ratio', 'Grad.Rate', 'perc.alumni']
X_train[col_per] = X_train[col_per]/100
X_test[col_per] = X_test[col_per]/100

####### 수치형 컬럼 간 상관관계 확인
# 상관관계를 확인할 컬럼만
# X_train.corr()

# Apps, Accept, Enroll, F.Undergrad 컬럼 간
# Top10perc와 Top25perc, PhD와 Terminal 컬럼 간
# 상관관계 높음
# Apps, Accept, Top25perc, Terminal 컬럼 삭제
col_del = ['Apps', 'Accept', 'Top25perc', 'Terminal']
X_train = X_train.drop(col_del, axis = 1)
X_test = X_test.drop(col_del, axis = 1)

###### STEP3-5. 데이터 분할
from sklearn.model_selection import train_test_split
# X_train과 y_train을 학습용(X_TRAIN, y_TRAIN)과 검증용(X_VAL, y_VAL)로 분할
X_TRAIN, X_VAL, y_TRAIN, y_VAL = train_test_split(X_train, y_train, random_state = 1234, test_size = 0.3, stratify = y_train)

# print(X_TRAIN.shape)
# print(X_VAL.shape)
# print(y_TRAIN.shape)
# print(y_VAL.shape)

###### STEP3-6. 인코딩
# 카테고리형 컬럼에 대하여 원-핫 인코딩 수행
# 없으므로 생략

###### STEP3-7. 스케일링
from sklearn.preprocessing import StandardScaler

# TRAIN 데이터 기준으로 스케일링함
scale = StandardScaler().fit(X_TRAIN)

# z-점수 표준화
X_TRAIN_STD = scale.transform(X_TRAIN)
X_VAL_STD = scale.transform(X_VAL)
X_TEST_STD = scale.transform(X_test)

###### STEP3-8. 입력 데이터셋 준비
import numpy as np
X_TRAIN = X_TRAIN_STD
X_VAL = X_VAL_STD

y_TRAIN = y_TRAIN.values.ravel()
y_VAL = y_VAL.values.ravel()

##### STEP4. 모델 학습
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

###### STEP4-1. random forest
# rf = RandomForestClassifier(n_estimators = 500,
#                            max_depth = 3,
#                            min_samples_leaf = 10,
#                            max_features = 'sqrt',
#                            random_state = 2022)
# model_rf = rf.fit(X_TRAIN, y_TRAIN)

###### STEP4-2. Bagging
# dtr = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 10)
# bag = BaggingClassifier(base_estimator = dtr,
#                        n_estimators = 500,
#                        random_state = 2022)
# model_bag = bag.fit(X_TRAIN, y_TRAIN)

###### STEP4-3. AdaBoost
ada = AdaBoostClassifier(n_estimators = 500,
                        learning_rate = 0.5,
                        random_state = 2022)
model_ada = ada.fit(X_TRAIN, y_TRAIN)

###### STEP4-4. 성능평가(기준:AUC)를 통한 모델 선정
# from sklearn.metrics import roc_curve, auc

# # 검증용 데이터셋을 통한 예측
# score_rf = model_rf.predict_proba(X_VAL)[:,1]
# score_bag = model_bag.predict_proba(X_VAL)[:,1]
# score_ada = model_ada.predict_proba(X_VAL)[:,1]

# # AUC 계산
# fpr, tpr, thresholds = roc_curve(y_VAL, score_rf, pos_label = 'Yes')
# auc_rf = auc(fpr, tpr)
# print(auc_rf)

# fpr, tpr, thresholds = roc_curve(y_VAL, score_bag, pos_label = 'Yes')
# auc_bag = auc(fpr, tpr)
# print(auc_bag)

# fpr, tpr, thresholds = roc_curve(y_VAL, score_ada, pos_label = 'Yes')
# auc_ada = auc(fpr, tpr)
# print(auc_ada)

##### STEP5. 결과 제출하기
###### 실제 시험에서 답 제출시에는 성능이 가장 우수한 모형 하나만 구현!
X_TEST = X_TEST_STD
y_score = model_ada.predict_proba(X_TEST)[:,1]

# 문제에서 요구하는 형태로 변환 필요
obj = {'ID' : ID,
       'prob_Private' : y_score}
result = pd.DataFrame(obj)

# 하위에 12345.csv 이름으로 저장하기
result.to_csv("12345.csv", index = False)


# ## (끝)
