#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 패키지 불러오기
import pandas as pd # Pandas
import numpy as np # NumPy


# ## PART 02) 데이터 다루기

# ## 2장. Pandas를 활용한 데이터 다루기

# ### 1절. Series와 DataFrame

# #### 1. Series(시리즈)

# ##### 가. 시리즈의 생성

# In[2]:


# Series객체 생성 : pd.Series(object, index)

# 1. object에 리스트(시퀀스)객체 이용
obj = ['이유리', '최민준', '김민지'] # 값 리스트
series_1 = pd.Series(obj) # 시리즈 생성
print(series_1)


# In[3]:


# 원하는 인덱스 지정
idx = ['A', 'B', 'C'] # 인덱스 리스트
series_2 = pd.Series(obj, index = idx) # 시리즈 생성
print(series_2)


# In[4]:


# 2. object에 딕셔너리객체 이용
obj = {'A':'이유리', 'B':'최민준', 'C':'김민지'} # 딕셔너리
series_3 = pd.Series(obj) # 시리즈 생성
print(series_3)


# ##### 나. 시리즈의 정보 확인

# In[5]:


# 시리즈 정보 확인 메소드
# .values
# 시리즈객체 내 값을 배열 형태로 반환
print(series_1.values)

# .index
# 시리즈객체 내 인덱스를 레이블 배열 형태로 반환
print(series_1.index)

# .dtypes
# 시리즈객체의 데이터 타입 확인
print(series_1.dtypes)

# .size
# 시리즈객체의 총 객체 수
print(series_1.size)


# In[6]:


# 시리즈객체의 인덱스 변경
print(series_1.index) # 변경 전

series_1.index = ['A', 'B', 'C'] # 변경
print(series_1.index) # 변경 후


# ##### 다. 시리즈의 인덱싱과 슬라이싱

# In[7]:


# 시리즈 생성
obj = [22, 32, 27, 18, 38, 19] # 값 리스트
idx = list('abcdef') # 인덱스 리스트
sr = pd.Series(obj, index = idx)
sr


# In[8]:


# 인덱싱
print(sr[4]) # 위치기반
print(sr['e']) # 레이블기반


# In[9]:


# 슬라이싱
print(sr[1:5]) # 위치기반
print(sr['b':'e']) # 레이블기반


# In[10]:


# 연속하지 않은 위치
idx = [0, 3, 4] # 위치 번호를 담은 리스트
sr[idx]


# In[11]:


# 연속하지 않은 레이블
lbl = ['a', 'd', 'e'] # 레이블을 담은 리스트
sr[lbl]


# ##### 라. 시리즈의 통계 메소드

# In[12]:


sr_1 = pd.Series(range(100)) # 시리즈 생성

# describe()
# 시리즈객체의 요약 통계량에 대한 정보
# 총 객체수, 평균, 표준편차, 최솟값, 제1사분위수, 중앙값, 제3사분위수, 최댓값, 데이터타입
sr_1.describe()


# In[13]:


# .count()
# 시리즈객체의 총 객체수
print(sr_1.count())

# .mean()
# 시리즈객체의 평균
print(sr_1.mean())

# .var()
# 시리즈객체의 분산
print(sr_1.var())

# .std()
# 시리즈객체의 표준편차
print(sr_1.std())

# .min()
# 시리즈객체의 최솟값
print(sr_1.min())

# .max()
# 시리즈객체의 최댓값
print(sr_1.max())

# .median()
# 시리즈객체의 중앙값
print(sr_1.median())

# .quantile()
# 시리즈객체의 q*100% 백분위수
print(sr_1.quantile(q = 0.25)) # 제1사분위수
print(sr_1.quantile()) # default = .5로 중앙값
print(sr_1.quantile(q = 0.75)) # 제3사분위수


# In[14]:


sr_2 = pd.Series(['a','b','b','b','c','c','d']) # 시리즈 생성
sr_2.describe() # 총 객체수, 유일값, 최빈값, 최빈값의빈도


# In[15]:


# .unique()
# 시리즈객체의 유일값을 1d-array로 반환
print(sr_2.unique())

# .value_counts()
# 시리즈객체의 유일값별 빈도수를 시리즈로 반환
print(sr_2.value_counts())

# .mode()
# 시리즈객체의 최빈값을 시리즈로 반환
print(sr_2.mode())


# #### 2. DataFrame(데이터프레임)

# ##### 가. 데이터프레임의 생성

# In[16]:


# DataFrame객체 생성 : pd.DataFrame(object, index, columns)

# 1. object에 동일한 길이의 리스트를 값으로 가지는 딕셔너리 이용
obj = {'이름' : ['이유리', '최민준', '김민지'],
       '전공' : ['경영학과', '컴퓨터공학과', '데이터과학과'],
       '성별' : ['여', '남', '여'],
       '나이' : [20, 22, 21]}

df_1 = pd.DataFrame(obj) # 데이터 프레임 생성
df_1


# In[17]:


# 2. object에 2차원 배열 이용
obj = np.array([['이유리', '경영학과', '여', 20],
                ['최민준', '컴퓨터공학과', '남', 22],
                ['김민지', '데이터과학과', '여', 21]])

df_2 = pd.DataFrame(obj) # 데이터 프레임 생성
df_2


# In[18]:


# 원하는 인덱스 지정
df_3 = pd.DataFrame(obj,
                    columns = ['Name', 'Major', 'Sex', 'Age'], # 열에 대한 인덱스
                    index = ['A', 'B', 'C'] # 행에 대한 인덱스
                   )
df_3


# In[19]:


# DataFrame객체의 행에 대한 인덱스 확인
print(df_1.index)

# DataFrame객체의 열에 대한 인덱스 확인
print(df_1.columns)

# DataFrame객체의 값 확인
print(df_1.values) # array로 반환됨


# ##### 나. 데이터프레임의 정보 확인

# In[20]:


# 데이터프레임 정보 확인 메소드
# DataFrame객체 생성 
obj = {'Name' : ['Olivia', 'Lucas', 'Sophia', 'Zoe', 'Ava', 'Elliot'],
       'Sex' : ['Female', 'Male', 'Female', 'Female', 'Female', 'Male'],
       'Age' : [22, 32, 27, 18, 38, 19]}
df = pd.DataFrame(obj) # 데이터 프레임 생성
print(df)

# .values
# 데이터프레임 내 값을 배열 형태로 반환
print(df.values)

# .index
# 데이터프레임 내 행 인덱스를 레이블 배열 형태로 반환
print(df.index)

# .columns
# 데이터프레임 내 열 인덱스를 레이블 배열 형태로 반환
print(df.columns)

# .dtypes
# 데이터프레임 내 변수(컬럼)별 데이터 타입 확인
print(df.dtypes)

# .shape
# 데이터프레임의 행, 열 길이 확인
print(df.shape)

# .info()
# 데이터프레임의 전반적인 요약 정보 확인
# 로우명, 컬럼명, 행 길이, 열 길이, 컬럼별 데이터 타입, 메모리 등 확인 가능
df.info()


# In[21]:


# .head(n=5)
# 데이터프레임의 상위 행을 반환(n은 반환할 행의 수)
df.head() # 상위 5(default)개의 행 반환


# In[22]:


df.head(1) # 상위 1개의 행 반환


# In[23]:


df.head(-3) # 상위 (전체행-3)개의 행 반환


# In[24]:


# .tail(n=5)
# 데이터프레임의 하위 행을 반환(n은 반환할 행의 수)
df.tail() # 하위 5(default)개의 행 반환


# In[25]:


df.tail(1) # 하위 1개의 행 반환


# In[26]:


df.tail(-3) # 하위 (전체행-3)개의 행 반환


# ##### 다. 데이터프레임의 인덱싱과 슬라이싱

# In[27]:


# 데이터프레임 생성
obj = {'Name' : ['Olivia', 'Lucas', 'Sophia', 'Zoe', 'Ava', 'Elliot'],
       'Sex' : ['Female', 'Male', 'Female', 'Female', 'Female', 'Male'],
       'Age' : [22, 32, 27, 18, 38, 19]}
idx = list('abcdef') # 인덱스 리스트

df = pd.DataFrame(obj, index = idx) 


# In[28]:


# Case 1. 열만 참조
# 데이터프레임명['컬럼명']
df['Name']


# In[29]:


# 데이터프레임명.컬럼명
df.Name


# In[30]:


# 여러 열을 추출
# 데이터프레임명[리스트]
colnm = ['Name', 'Age'] # 추출할 열의 컬럼명
df[colnm] # Name과 Age를 열(시리즈)로 가지는 데이터프레임


# In[31]:


# Case 2. 행만 참조
# iloc은 위치 기반으로, 데이터프레임명.iloc[행위치인덱스]

# 2번 째(표 기준)행의 값과 원래의 컬럼명을 레이블 인덱스으로 가지는 시리즈
df.iloc[1]


# In[32]:


# 2~5번 째(표 기준) 행을 가지는 데이터프레임
df.iloc[1:5]


# In[33]:


# 연속하지 않은 위치일 경우에도 리스트로 행을 참조
idx = [1, 3] 
df.iloc[idx] 


# In[34]:


# loc은 레이블 기반이며, 데이터프레임명.loc[행레이블인덱스]
# 2번 째(표 기준)행의 값과 원래의 컬럼명을 레이블 인덱스으로 가지는 시리즈
df.loc['b']


# In[35]:


# 2~5번 째(표 기준) 행을 가지는 데이터프레임
df.loc['b':'e']


# In[36]:


# 대부분의 레이블 인덱스는 연속하지 않음
rownm = ['b', 'd'] # 리스트로 행을 참조
df.loc[rownm] # 2, 4번째(표 기준) 행을 가지는 데이터프레임


# In[37]:


# Case 3. 인덱서 없이 행과 열을 동시에 참조하는 방법
# i) 하나의 열에 대해 인덱싱
df['Age'] #Age 열 참조


# In[38]:


# ii) 행에 대해 인덱싱
# 시리즈가 되기 때문에 가능해짐
df['Age']['a'] #Age 열과 a 행 참조


# In[37]:


df['Age'][0] #Age 열과 0번 위치 행 참조


# In[38]:


# ii) loc를 통한 방법 : 2d-array에서의 방법에 숫자 대신 레이블을 이용
# 행 인덱스(로우명)이 b이고 열 인덱스(컬럼명)이 Name
df.loc['b', 'Name']


# In[40]:


# 대부분 행과 열의 레이블 인덱스(특히 열)은 연속되지 않아 리스트를 사용
# 행 : b, d, e
# 열 : Name Sex
rownm = ['b', 'd', 'e']
colnm = ['Name', 'Sex']
df.loc[rownm, colnm]


# ##### 라. 데이터프레임의 통계 메소드

# In[41]:


# 데이터프레임 생성
obj = {'Name' : ['Olivia', 'Lucas', 'Sophia', 'Zoe', 'Ava', 'Elliot'],
       'Sex' : ['Female', 'Male', 'Female', 'Female', 'Female', 'Male'],
       'Age' : [22, 32, 27, 18, 38, 19],
       'Score' : [100, 95, 60, 77, 83, 84]
       }

df = pd.DataFrame(obj) # 데이터 프레임 생성


# In[42]:


df.describe() # 컬럼별 요약 통계량에 대한 정보(float, int인 경우만)


# 

# ### 2절. 데이터 입출력

# 코드 없음

# 

# ### 3절. 데이터 정렬 및 순위

# #### 1. 데이터 정렬

# In[43]:


import pandas as pd


# In[44]:


# .sort_values
# 데이터 프레임 생성
obj = {'Name' : ['Olivia', 'Lucas', 'Sophia', 'Zoe', 'Ava', 'Elliot'],
       'Age' : [22, 22, 27, 18, 18, 19],
       'Score' : [100, 95, 60, 77, 83, 84]
       }
df = pd.DataFrame(obj)

# Score 열에 대하여 데이터프레임 오름차순 정렬
df.sort_values('Score')


# In[45]:


# Age와 Score 열에 대하여 데이터프레임 정렬
# Age에 대해 오름차순으로 정렬되고, 동일한 Age에 대해 Score가 내림차순 정렬됨
df.sort_values('Score', ascending = False)


# In[48]:


# Age와 Score 열에 대하여 데이터프레임 정렬
# Age에 대해 오름차순으로 정렬되고, 동일한 Age에 대해 Score가 내림차순 정렬됨
df.sort_values(['Age', 'Score'], ascending = [True, False])


# In[46]:


# .sort_index
# 데이터 프레임 생성
rownm = ['Olivia', 'Lucas', 'Sophia', 'Zoe', 'Ava', 'Elliot']

obj = {'Score' : [100, 95, 60, 77, 83, 84],
       'Age' : [22, 22, 27, 18, 18, 19],
       }
df = pd.DataFrame(obj, index = rownm)


# In[47]:


# 행 인덱스에 대해 오름차순으로 정렬됨
df.sort_index()


# In[48]:


# 열 인덱스에 대해 오름차순으로 정렬됨
df.sort_index(axis = 1)


# #### 2. 데이터 순위

# In[49]:


# .rank

# 데이터 프레임 생성
obj = {'Score' : [100, 95, 60, 77, 83, 85, 90, 90, 88, 75, 90, 54, 48, 84, 73],
       'Age' : [22, 22, 27, 18, 18, 19, 24, 26, 30, 27, 25, 21, 20, 17, 20]
       }
df = pd.DataFrame(obj)


# In[50]:


# Score에 대해 순위 생성
# 결과 비교가 용이하도록 정렬을 먼저 수행
# inplace = True로 원본 데이터프레임을 바로 변경
df.sort_values('Score', ascending = False, inplace = True)
df.head()


# In[51]:


# case1. 동점자 평균 순위
df['rank_avg1'] = df['Score'].rank()

# case2. 동점자 평균 순위(가장 큰 값이 1위)
# Score가 90인 경우가 3개로, 3~5위의 평균인 4위
df['rank_avg2'] = df['Score'].rank(ascending = False)

# case3. 동점자 가장 낮은 순위(가장 큰 값이 1위)
# Score가 90인 경우가 3개로, 3~5위 중 가장 낮은 3위 
df['rank_min'] = df['Score'].rank(method = 'min', ascending = False)

# case4. 동점자 가장 높은 순위(가장 큰 값이 1위)
# Score가 90인 경우가 3개로, 3~5위 중 가장 높은 5위 
df['rank_max'] = df['Score'].rank(method = 'max', ascending = False)
print(df)


# ----- 
