#!/usr/bin/env python
# coding: utf-8

# ## PART 01) 데이터 다루기

# ## 3장. 데이터 변환

# ### 1절. 파생변수 생성

# #### 1. 컬럼 추가

# In[1]:


# 사이킷런 내의 datasets 모듈 내 iris 데이터 호출

from sklearn.datasets import load_iris
iris = load_iris()
# iris는 딕셔너리형과 유사한 구조로 key와 value를 가지고 있음
# 'data' : 2d-array로 꽃 받침과 꽃잎의 길이, 너비(단위; cm)
# 'target': 1d-array로 품종(숫자변환)
# 'target_names': 1d-array로 품종(문자)
# 'feature_names' : 2d-array로 'data'의 컬럼명

import pandas as pd # pandas 패키지 호출
data = pd.DataFrame(iris.data, columns = iris.feature_names)
data


# In[2]:


# 데이터 구조 확인 : 150개의 행과 4개의 열을 가짐
data.info()


# In[3]:


# 150개 개체들의 고유 번호에 해당하는 'ID' 컬럼을 새로 생성
# 1. 대괄호([ ])를 이용한 방법
data['ID'] = range(150)
data.head() # 상위 5개 행 출력


# In[4]:


# 2. insert 메소드를 이용한 방법
data = pd.DataFrame(iris.data, columns = iris.feature_names) # 다시 생성
data.insert(0, column = 'ID', value = range(150))
data.head() # 상위 5개 행 출력


# #### 2. 컬럼 삭제

# In[5]:


# .drop() 메소드를 통한 컬럼 삭제
data2 = data.drop('ID', axis = 1)
data2.head() # 상위 5개 행 출력


# #### 3. 구간화(binning)

# In[6]:


# .cut()
# 5개의 구간으로 분할
pd.cut(data['sepal length (cm)'], bins = 5)


# In[8]:


# 구간 (4, 5], (5, 6], (6, 7], (7, 8]으로 분할
pd.cut(data['sepal length (cm)'], bins = [4, 5, 6, 7, 8])


# In[7]:


# 컬럼 'sepal length (cm)'의 요약 통계량에 대한 정보
# .qcut()
# 4분위수마다 분할
pd.qcut(data['sepal length (cm)'], q = 4)


# In[8]:


# 4분위수 확인
data['sepal length (cm)'].quantile([.25, .5, .75])


# In[9]:


# 10분위수마다 분할
pd.qcut(data['sepal length (cm)'], q = 10)


# In[10]:


# 10분위수 확인(0과 1은 최솟값, 최댓값)
data['sepal length (cm)'].quantile([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])


# 

# ### 2절. 데이터 인코딩

# #### 1. 라벨 인코딩(Label Encoding)

# In[11]:


import pandas as pd

# 사이킷런 내의 preprocessing 모듈 내 LabelEncoder 클래스 호출
from sklearn.preprocessing import LabelEncoder

# 데이터프레임 생성
obj = {'Class' : ['A', 'B', 'A', 'C', 'E', 'D', 'D', 'A', 'F']}
df = pd.DataFrame(obj)

# 라벨 인코딩
encoder = LabelEncoder() # 라벨 인코더객체 생성
encoder.fit(df.Class) # 적합
labels = encoder.transform(df.Class) # 변환

# 인코딩 결과, 변환된 숫자 카테고리 값
print(labels)

# 숫자 값에 대응되는 원본 레이블
print(encoder.classes_)


# In[12]:


# 디코딩(즉, 다시 원본 문자열로 되돌림)
print(encoder.inverse_transform(labels))


# In[14]:


# numpy.select()를 이용해 라벨 인코딩과 동일한 결과를 만드는 방법
import numpy as np

# 조건 목록 생성
conditionlist = [(df['Class'] == 'A'),
                 (df['Class'] == 'B'),
                 (df['Class'] == 'C'),
                 (df['Class'] == 'D'),
                 (df['Class'] == 'E'),
                 (df['Class'] == 'F')]

# 조건과 매칭할 선택 목록 생성(0~5)
choicelist  = list(range(6))

# 결과
np.select(condlist = conditionlist, choicelist = choicelist)


# In[15]:


# 시리즈객체.map()을 이용해 라벨인코딩과 동일한 결과를 만드는 방법
df['Class'].map(arg = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5})


# #### 2. 원-핫 인코딩(One-Hot Encoding)

# In[16]:


# 사이킷런 OneHotEncoder 클래스를 활용한 방법
# 사이킷런 내의 preprocessing 모듈 내 LabelEncoder, OneHotEncoder 클래스 호출
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 데이터 프레임 생성
obj = {'Class' : ['A', 'B', 'A', 'C', 'E', 'D', 'D', 'A', 'F']}
df = pd.DataFrame(obj)

# 라벨 인코딩 먼저 수행
encoder = LabelEncoder()
encoder.fit(df['Class'])

# 2차원 레이블 변환
labels = encoder.transform(df['Class']).reshape(-1,1)
 
# 원-핫 인코딩
# sparse = False 옵션은 결과를 보통의 array 형태로 반환하기 위함이다.
oh_encoder = OneHotEncoder(sparse = False)
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)

# 결과
oh_labels


# In[17]:


# 판다스 get_dummies() 함수를 활용한 방법
obj = {'Class' : ['A', 'B', 'A', 'C', 'E', 'D', 'D', 'A', 'F']}
df = pd.DataFrame(obj)
pd.get_dummies(df)# 원핫인코딩 실행


# In[18]:


# 첫 번째 고유 특징을 제거해 k-1개의 변수를 생성
pd.get_dummies(df, drop_first = True)


# In[19]:


# numpy.where()를 이용해 라벨 인코딩과 동일한 결과를 만드는 방법
df['Class_B'] = np.where(df['Class'] == 'B', 1, 0)
df['Class_C'] = np.where(df['Class'] == 'C', 1, 0)
df['Class_D'] = np.where(df['Class'] == 'D', 1, 0)
df['Class_E'] = np.where(df['Class'] == 'E', 1, 0)
df['Class_F'] = np.where(df['Class'] == 'F', 1, 0)
df


# In[20]:


# loc 인덱서와 부울 인덱싱을 이용한 방법
obj = {'Class' : ['A', 'B', 'A', 'C', 'E', 'D', 'D', 'A', 'F']}
df = pd.DataFrame(obj)

# df의 Class열이 B이면, Class_B 열에 1을 추가 
df.loc[df['Class'] == 'B', 'Class_B'] = 1
# df의 Class열이 B가 아니면, Class_B 열에 0을 추가 
df.loc[df['Class'] != 'B', 'Class_B'] = 0 

# df의 Class열이 C이면, Class_C 열에 1을 추가 
df.loc[df['Class'] == 'C', 'Class_C'] = 1
# df의 Class열이 C가 아니면, Class_C 열에 0을 추가 
df.loc[df['Class'] != 'C', 'Class_C'] = 0 

# df의 Class열이 D면, Class_D열에 1을 추가 
df.loc[df['Class'] == 'D', 'Class_D'] = 1
# df의 Class열이 D가 아니면, Class_D열에 0을 추가 
df.loc[df['Class'] != 'D', 'Class_D'] = 0 

# df의 Class열이 E면, Class_E열에 1을 추가 
df.loc[df['Class'] == 'E', 'Class_E'] = 1
# df의 Class열이 E가 아니면, Class_E열에 0을 추가 
df.loc[df['Class'] != 'E', 'Class_E'] = 0 

# df의 Class열이 F면, Class_F열에 1을 추가 
df.loc[df['Class'] == 'F', 'Class_F'] = 1
# df의 Class열이 F가 아니면, Class_F열에 0을 추가 
df.loc[df['Class'] != 'F', 'Class_F'] = 0 

df # 확인


# In[21]:


# 해당 예제는 구간화와 파생 변수 생성, 인코딩이 모두 필요함
# 패키지, 모듈, 함수 호출
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 데이터프레임 생성
obj = {'student_id' : ['s1', 's2', 's3', 's4', 's5', 's6'],
       'score' : [55, 90, 85, 71, 63, 99]}
df = pd.DataFrame(obj)
print(df)

# 구간화한 파생변수 grade 열 생성
df['grade'] = pd.cut(df['score'],
                     bins = [0, 60, 70, 80, 90, 100],
                     labels = ['가', '양', '미', '우', '수'])

print(df)

# sklearn을 활용한 라벨 인코딩
lb_encoder = LabelEncoder()
lb_encoder.fit(df['grade'])
lb_labels = lb_encoder.transform(df['grade'])
print(lb_labels) 
print(lb_encoder.classes_) # 숫자 값에 대응되는 원본 레이블

# sklearn을 활용한 원-핫 인코딩
lb_labels_2d = np.array(df['grade']).reshape(-1, 1) # 2d 라벨
lb_labels_2d 

oh_encoder = OneHotEncoder(sparse = False)
oh_encoder.fit(lb_labels_2d)

oh_labels = oh_encoder.transform(lb_labels_2d)
print(oh_labels)


# 

# ### 3절. 데이터 스케일링

# #### 1. 표준화(standardization)

# In[27]:


# 패키지 설치
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# iris 데이터셋 호출
iris = load_iris()
data = pd.DataFrame(iris.data, columns = iris.feature_names) # 데이터프레임 변환

# 표준 스케일러객체 생성
std_scale = StandardScaler()

# 표준화 데이터로 변환
data_std = std_scale.fit_transform(data)
data_std = pd.DataFrame(data_std, columns = iris.feature_names) # 데이터프레임 변환
data_std.head() # 확인


# #### 2. 정규화(Normalization)

# In[22]:


from sklearn.preprocessing import MinMaxScaler

# iris 데이터셋 호출
iris = load_iris()
data = pd.DataFrame(iris.data, columns = iris.feature_names) # 데이터프레임 변환

# 최소최대 스케일러객체 생성
mm_scale = MinMaxScaler()

# 정규화 데이터로 변환
data_std = mm_scale.fit_transform(data)
data_std = pd.DataFrame(data_std, columns = iris.feature_names) # 데이터프레임 변환
print(data_std) # 확인


# 

# ### 4절. 데이터 축소

# #### 1. 주성분 분석

# ##### 가. 주성분분석의 개념

# 코드 없음

# ##### 나. 주성분의 선택

# 코드 없음

# ##### 다. 파이썬을 이용한 주성분분석

# In[25]:


# 패키지 불러오기
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 제공된 USArrests 데이터 불러오기
# 주가 행 인덱스 레이블이 되도록 index_col = 0
df = pd.read_csv("data/예제/USArrests_rownames.csv", index_col = 0)
df.head() # 확인


# In[26]:


# 주성분 분석 전에 표준화(스케일링)하기
# fit_transform은 한 번에 fit과 transform을 수행
x = StandardScaler().fit_transform(df) # x객체에 df 전체를 표준화한 데이터 할당

# PCA객체 생성(주성분의 수를 전체 컬럼 수로)
pca = PCA(n_components = 4)

# 적합후 변환(배열형태)
pca_arr = pca.fit_transform(x)

# 주성분에 따른 분산 기여율과 누적 기여율
result = {"분산 기여율" : pca.explained_variance_ratio_, # 분산 기여율
          "누적 기여율" : pca.explained_variance_ratio_.cumsum()} # 누적 기여율

# 결과를 데이터프레임 형태로 변환, 전치하여 컬럼명이 주성분이 되게 함
result = pd.DataFrame(result,
                      index = ['PCA1', 'PCA2', 'PCA3', 'PCA4']).T
print(result) # 주성분의 수를 2개로 정함


# In[27]:


# 주성분 분석 결과
pca = PCA(n_components = 2) # PCA객체 생성
pca_arr = pca.fit_transform(x) # 적합후 변환(배열형태)
df_pca = pd.DataFrame(pca_arr, columns = ['PCA1', 'PCA2']) # 데이터프레임으로 변환
df_pca.head() # 확인


# ----
