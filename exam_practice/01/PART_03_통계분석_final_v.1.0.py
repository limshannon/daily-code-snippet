#!/usr/bin/env python
# coding: utf-8

# ### PART 03) 통계분석

# ## 1장. 상관분석

# ### 1절. SciPy를 활용한 상관분석

# In[1]:


# 데이터 호출한 후 데이터프레임으로 변환
import pandas as pd
from sklearn.datasets import load_diabetes 

diabetes = load_diabetes()
data = pd.DataFrame(diabetes.data, columns = diabetes.feature_names)
data.info() # 확인


# In[3]:


# scipy.stats.pearsonr
from scipy.stats import pearsonr
pearsonr(x = data['age'], y = data['bmi'])


# In[2]:


# 단순한 상관계수의 산출은 데이터프레임객체.corr()로도 가능
data[['age', 'bmi']].corr()


# In[3]:


# scipy.stats.spearmanr
from scipy.stats import spearmanr
spearmanr(a = data['sex'], b = data['bmi'])


# In[4]:


# 단순한 상관계수의 산출은 데이터프레임객체.corr()로 가능
# corr(method = 'spearman')은 스피어만 상관계수를 산출함
data[['sex', 'bmi']].corr(method = 'spearman')


# --- 

# ## 2장. 회귀분석

# ### 1절. 선형 회귀분석

# #### 1. SciPy를 활용한 단순 선형 회귀분석

# In[5]:


# 'target’ 컬럼 호출
target = diabetes.target

# 단순 선형회귀 모델 생성
# scipy.stats.linregress()
from scipy.stats import linregress
model = linregress(x = data['bmi'], y = target)
print(model) # 전체 결과


# In[6]:


# 독립변수에 대한 추정된 회귀계수(beta1)
print(model.slope)

# 상수항에 대한 추정된 회귀계수(beta0)
print(model.intercept)

# beta1에 대한 통계적 유의성(p-value)
print(model.pvalue)


# In[7]:


# 결정계수(모형의 설명력)
print(model.rvalue)


# #### 2. 사이킷런을 활용한 선형 회귀분석

# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


colnm = ['bmi', 'bp', 's1', 's2', 's3'] # 컬럼명 리스트
X = data[colnm]
y = target


# In[10]:


# 선형회귀 객체 생성
model = LinearRegression()
# 선형회귀 적합
model.fit(X = X, y = y)


# In[11]:


# 독립변수들에 대한 추정 회귀 계수들
print(model.coef_)


# In[12]:


# 절편항에 대한 추정 회귀 계수 
print(model.intercept_)


# In[13]:


# 결정계수
model.score(X = X, y = y)


# #### 3. 정규화 선형 회귀분석

# ##### 가. 릿지 회귀(Ridge Regression)

# In[18]:


from sklearn.linear_model import Ridge
colnm = ['bmi', 'bp', 's1', 's2', 's3'] # 컬럼명 리스트
X = data[colnm]
y = target

# 릿지회귀객체 생성
model = Ridge(alpha = 0.1)

# 적합
model.fit(X = X, y = y)


# In[20]:


# 독립변수들에 대한 추정 회귀 계수들
print(model.intercept_)
print(model.coef_)


# ##### 나. 라쏘 회귀(Lasso Regression)

# In[21]:


from sklearn.linear_model import Lasso
colnm = ['bmi', 'bp', 's1', 's2', 's3'] # 컬럼명 리스트
X = data[colnm]
y = target

# 라쏘회귀객체 생성
model = Lasso(alpha = 0.5)

# 적합
model.fit(X = X, y = y)


# In[22]:


# 독립변수들에 대한 추정 회귀 계수들
print(model.intercept_)
print(model.coef_)


# In[24]:


# 패키지 및 데이터셋, 클래스 호출
import pandas as pd
from sklearn.datasets import load_diabetes 
from sklearn.linear_model import Lasso

# diabetes 데이터셋 호출 후 데이터프레임으로 변환
diabetes = load_diabetes() 
data = pd.DataFrame(diabetes.data, columns = diabetes.feature_names) 
target = pd.Series(diabetes.target, name = 'target')
df = pd.concat([data, target], axis = 1) # 데이터프레임과 시리즈를 열 결합

# 데이터 분할
colnm = ['bmi', 'bp', 's1', 's2', 's3'] # 컬럼명 리스트
X_train = df[colnm].loc[:310] # 0~309번 행과 ‘bmi’,‘bp’,‘s1’,‘s2’,‘s3’ 컬럼
X_test = df[colnm].loc[310:] # 310~441번 행 ‘bmi’,‘bp’,‘s1’,‘s2’,‘s3’ 컬럼
y_train = df['target'].loc[:310] # 0~309번 행과 'target’ 컬럼

# 라쏘회귀객체 생성
model = Lasso(alpha = 0.5) 
model.fit(X = X_train, y = y_train) # X_train과 y_train으로 라쏘 회귀모형 적합

# X_test를 통해 새로우 'target' 변수를 예측
target = model.predict(X_test)
target = pd.Series(target, name = 'target') # array -> series
print(target)


# # (끝)
