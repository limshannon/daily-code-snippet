#!/usr/bin/env python
# coding: utf-8

# ## 모의고사 1회

# ### (작업형1) 

# #### 1. iris  데이터셋을 불러와 Sepal.Width 컬럼에 대해 Sepal.Width의 평균값을 기준으로 3배 표준편차 이상으로 떨어진 값들의 합을 구하여라.

# In[1]:


#### 모의고사 01회 01 Solution
import pandas as pd
exam1 = pd.read_csv('data/모의고사/01회/iris.csv')

##### sepal_width 별도 저장
sepal_width = exam1['Sepal.Width']

##### sepal_width 평균 기준 3배 표준편차 이상 떨어진 데이터추출
# sepal_width의 평균
avg = sepal_width.mean()

# sepal_length의 표준편차
sd = sepal_width.std()

# 상한과 하한
upp = avg + 3 * sd
low = avg - 3 * sd

##### sepal_width 평균 기준 3배 표준편차 이상 벗어날 조건
# 하한보다 작고 상한보다 큼
cond = (sepal_width < low) | (sepal_width > upp)

##### 결과를 result에 할당
result = sepal_width[cond].sum() # 떨어진 값들의 합

##### 결과 출력
print(result)


# --- 

# #### 2. mtcars1 데이터셋을 불러와 disp 컬럼에 대해서 순위를 부여한 후, 1위부터 20위까지의 값들의 표준편차를 구하고 소수점 셋째 자리에서 반올림하여 나타내어라. (단, 동점은 동일한 순위를 부여하되 상위 등수를 기준으로 하며 최댓값을 1위로 함)

# In[2]:


#### 모의고사 01회 02 Solution
import pandas as pd
exam2 = pd.read_csv('data/모의고사/01회/mtcars1.csv')

# disp 컬럼 별도 저장
disp = exam2['disp']

# disp 순위 부여
rank = disp.rank(method = 'min', ascending = False)

# 1위부터 20위까지의 값 
rank20 = disp[rank <= 20]

##### 결과를 result에 할당
result = round(rank20.std(), 2)

##### 결과 출력
print(result)


# --- 

# #### 3. Cars93 데이터셋을 불러와 전체 레코드 수, 결측치가 있는 컬럼의 수, 전체 결측치 수, 결측치가 10개 이상인 컬럼들의 결측치가 있는 레코드만 삭제한 후의 전체 레코드의 수와 두 개 이상의 컬럼이 동시에 결측인 레코드의 행번호들의 합을 구한 후 모두 합하여라.

# In[3]:


#### 모의고사 01회차 03 Solution
import pandas as pd
exam3 = pd.read_csv('data/모의고사/01회/Cars93.csv')

##### case1. 전체 레코드 수
case1 = exam3.shape[0]

##### case2. 결측치가 있는 컬럼의 수
case2 = sum(exam3.isna().sum() != 0)

##### case3. 전체 결측치 수
case3 = sum(exam3.isna().sum())

##### case4. 결측치가 10개 이상인 컬럼들의 결측치가 있는 레코드만 삭제한 후의 전체 레코드의 수
# 결측치의 수가 10개가 이상인 컬럼명을 colnm_10over에 할당
colnm_10over = exam3.columns[exam3.isna().sum() > 10]

# 그 중에서 결측치가 없는 경우의 전체 레코드 수
sub1 = exam3[colnm_10over].copy()
case4 = len(sub1.dropna())

##### case5. 두 개 이상의 컬럼이 동시에 결측인 레코드의 행 번호들의 합
# 결측치의 수가 2개가 이상인 행 인덱스를 rownm_2over에 할당
rownm_2over = exam3.index[exam3.isna().sum(axis = 1) >= 2]

# 행 번호를 리스트로 반환한 후 합함
sub2 = list(rownm_2over)
case5 = sum(sub2)

##### 결과를 result에 할당
result = case1 + case2 + case3 + case4 + case5

##### 결과 출력
print(result)


# ---

# ### (작업형2) 

# ### 1. 아래는 타이타닉호의 탑승자들의 생존과 관련한 데이터이다. 주어진 데이터를 이용하여 예측 모형을 만들고 아래에 따라 CSV 파일을 생성하시오.

# In[4]:


#### 모의고사 01회차_작업형2 Solution

##### STEP1. 데이터셋 불러오기
import pandas as pd
X_train = pd.read_csv('data/모의고사/01회/titanic3_X_train.csv')
X_test = pd.read_csv('data/모의고사/01회/titanic3_X_test.csv')
y_train = pd.read_csv('data/모의고사/01회/titanic3_y_train.csv')


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
# ID 컬럼은 탑승자에 대한 고유 정보로 key 역할로 모델에는 불필요함
# 결과 제출 시에는 X_test의 ID 컬럼이 필요하기 때문에 별도 저장
ID = X_test['ID'].copy()

# name은 텍스트 전처리 등의 방법으로 분석 가능하기도 하지만 편의상 제외
# 데이터들에서 ID, name 컬럼 삭제
X_train = X_train.drop(columns = ['ID', 'name'])
X_test = X_test.drop(columns = ['ID', 'name'])
y_train = y_train.drop(columns = 'ID')


# In[9]:


###### STEP3-2. 결측치 처리
# 결측치 확인
X_train.isna().sum()


# In[10]:


X_test.isna().sum()


# In[11]:


####### age 컬럼(train 157, test 106 결측)
# age는 탑승자의 나이를 의미하고 survived와 상관관계가 낮으므로 컬럼을 삭제

# 결측일 조건
cond_na = X_train['age'].isna()

# 피어슨 상관계수
from scipy.stats import pearsonr
pearsonr(y_train['survived'][~cond_na], X_train['age'][~cond_na])


# In[12]:


# 완료 후 Age 컬럼을 삭제
X_train = X_train.drop('age', axis = 1)
X_test = X_test.drop('age', axis = 1)


# In[13]:


####### fare 컬럼(train 1 결측)
# fare는 티켓요금을 의미하고 train에만 결측치가 1개 존재하므로 레코드를 삭제함

# 결측일 조건
cond_na = X_train['fare'].isna()

# 행 삭제
X_train = X_train[~cond_na]
y_train = y_train[~cond_na]


# In[14]:


####### cabin 컬럼(train 614, test 400 결측)
# cabin는 선실번호를 의미하고 train은 레코드의 78%, test는 레코드의 76%가 결측이므로 컬럼을 삭제

# cabin 컬럼을 삭제
X_train = X_train.drop('cabin', axis = 1)
X_test = X_test.drop('cabin', axis = 1)


# In[15]:


####### embarked 컬럼(train 1, test 1 결측)
# embarked는 탑승한 곳을 의미하고 범주형으로 최다빈도를 가지는 범주로 대체함

# 최다빈도
top = X_train['embarked'].value_counts().idxmax()

# 대치
X_train['embarked'] = X_train['embarked'].fillna(top)
X_test['embarked'] = X_test['embarked'].fillna(top)


# In[16]:


###### STEP3-3. 카테고리형 컬럼 전처리
# 문자열(object) 컬럼들의 유일값 수 확인
print(X_train.select_dtypes('object').nunique())
print(X_test.select_dtypes('object').nunique())


# In[17]:


####### sex 컬럼
# 여성에 대한 일부 카테고리가 'F'로 되어있음
X_train['sex'].value_counts()


# In[18]:


X_test['sex'].value_counts()


# In[19]:


# train, test 모두 'F'를 'female'로 통일
X_train['sex'] = X_train['sex'].map({'male':'male', 'female':'female', 'F':'female'})
X_test['sex'] = X_test['sex'].map({'male':'male', 'female':'female', 'F':'female'})


# In[20]:


####### ticket 컬럼
# 대다수가 중복되지 않으므로 컬럼을 삭제하는 것으로 결정
X_train = X_train.drop('ticket', axis = 1)
X_test = X_test.drop('ticket', axis = 1)


# In[21]:


###### STEP3-4. 수치형 컬럼 전처리
####### pclass 컬럼
# 수치형으로 인식되지만 1,2,3등석 정보를 각 1,2,3으로 저장한 것으로
# 카테고리의 의미를 가지는 컬럼
# dtype 변경 후 파생변수 pclass_gp에 할당하고 기존 컬럼 삭제
X_train['pclass_gp'] = X_train['pclass'].astype('object')
X_test['pclass_gp'] = X_test['pclass'].astype('object')

# 완료 후 삭제
X_train = X_train.drop('pclass', axis = 1)
X_test = X_test.drop('pclass', axis = 1)

####### sibsp, parch 컬럼
# sibsp는 동승한 형제 또는 배우자의 수, parch는 동승한 부모 또는 자녀의 수이므로
# 두 컬럼을 합한 파생변수 fam을 생성하고 이는 동승한 가족 인원을 의미
X_train['fam'] = X_train['sibsp'] + X_train['parch']
X_test['fam'] = X_test['sibsp'] + X_test['parch']

# 완료 후 삭제
X_train = X_train.drop(['sibsp', 'parch'], axis = 1)
X_test = X_test.drop(['sibsp', 'parch'], axis = 1)


# In[22]:


###### STEP3-5. 데이터 분할
from sklearn.model_selection import train_test_split

# X_train과 y_train을 학습용(X_TRAIN, y_TRAIN)과 검증용(X_VAL, y_VAL)로 분할
X_TRAIN, X_VAL, y_TRAIN, y_VAL = train_test_split(X_train, y_train, random_state = 1234, test_size = 0.1)

# 분할 후 shape 확인
print(X_TRAIN.shape)
print(X_VAL.shape)
print(y_TRAIN.shape)
print(y_VAL.shape)


# In[23]:


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


# In[24]:


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


# In[25]:


###### STEP3-8. 입력 데이터셋 준비
import numpy as np

# 인코딩과 스케일링된 넘파이 배열 연결
X_TRAIN = np.concatenate([X_TRAIN_OH, X_TRAIN_STD], axis = 1)
X_VAL = np.concatenate([X_VAL_OH, X_VAL_STD], axis = 1)

# 1차원 넘파이 배열로 평탄화
y_TRAIN = y_TRAIN.values.ravel()
y_VAL = y_VAL.values.ravel()


# In[26]:


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


# In[27]:


###### STEP4-4. 성능평가(기준:f1-score)를 통한 모델 선정
from sklearn.metrics import f1_score

# 검증용 데이터셋을 통한 예측
pred_rf = model_rf.predict(X_VAL)
pred_xgb = model_xgb.predict(X_VAL)
pred_lgb = model_lgb.predict(X_VAL)

# f1-score 계산
f1_rf = f1_score(y_VAL, pred_rf)
print(f1_rf)

f1_xgb = f1_score(y_VAL, pred_xgb)
print(f1_xgb)

f1_lgb = f1_score(y_VAL, pred_lgb)
print(f1_lgb)


# In[28]:


##### STEP5. 결과 제출하기
###### 실제 시험에서 답 제출시에는 성능이 가장 우수한 모형 하나만 구현!
X_TEST = np.concatenate([X_TEST_OH, X_TEST_STD], axis = 1)
y_pred = model_rf.predict(X_TEST)

# 문제에서 요구하는 형태로 변환 필요
obj = {'ID' : ID,
       'survived' : y_pred}
result = pd.DataFrame(obj)

# 하위에 12345.csv 이름으로 저장하기
result.to_csv("12345.csv", index = False)


# In[29]:


##### STEP6. 채점 모델 평가(번외)
# 실제값
actual = pd.read_csv('data/모의고사/01회/titanic3_y_test.csv', encoding = 'cp949')
actual = actual['survived'].ravel()

# 채점 기준이 될 성과지표 값
f1_score(actual, y_pred)


# # (끝)
