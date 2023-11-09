#!/usr/bin/env python
# coding: utf-8

# ## PART 01) 데이터 다루기

# ## 4장. 데이터 결합 및 요약

# ### 1절. 데이터 결합

# #### 1. 데이터 붙이기

# In[1]:


import pandas as pd

# 데이터프레임1 생성
obj1 = {'student_id' : ['s1', 's2', 's3', 's4', 's5', 's6'],
        'score' : [55, 90, 85, 71, 63, 99]}
df1 = pd.DataFrame(obj1)
print(df1)

# 데이터프레임2 생성
obj2 = {'student_id' : ['t1', 't2', 't3', 't4', 't5', 't6'],
        'score' : [65, 99, 87, 75, 57, 88]}
df2 = pd.DataFrame(obj2)
print(df2)


# In[2]:


# pandas.concat()
# 두 데이터프레임을 행을 기준으로 붙임
pd.concat([df1, df2])


# In[3]:


# 두 데이터프레임을 열을 기준으로 붙임
pd.concat([df1, df2], axis = 1)


# #### 2. 데이터 병합

# In[4]:


import pandas as pd

# 데이터프레임1 생성
obj1 = {'student_id' : ['s3', 's4', 's5', 's6'],
        'stat_score' : [85, 71, 63, 99]}
df1 = pd.DataFrame(obj1)
print(df1)

# 데이터프레임2 생성
obj2 = {'student_id' : ['s1', 's2', 's3', 's4'],
        'math_score' : [65, 99, 87, 75]}
df2 = pd.DataFrame(obj2)
print(df2)


# In[5]:


# 데이터프레임객체.merge()
# 두 데이터프레임 병합

# Case1. 병합유형(how) : inner(default)
# 두 데이터프레임의 공통 학생들의 점수만이 합쳐짐
df1.merge(df2, on = 'student_id')


# In[6]:


# Case2. 병합유형(how) : outer
# 두 데이터프레임의 모든 학생들의 점수가 합쳐짐
# 점수가 없는 과목은 NaN을 반환
# 학생 순서는 왼쪽 데이터 프레임인(df1)에 있는 student_id 먼저 나옴
df1.merge(df2, how = 'outer', on = 'student_id')


# In[7]:


# Case3. 병합유형(how) : left
# 왼쪽 데이터 프레임(df1)에 있는 학생들의 점수들만 반환
# 점수가 없는 과목은 NaN을 반환
df1.merge(df2, how = 'left', on = 'student_id')


# In[8]:


# Case4. 병합유형(how) : right
# 오른쪽 데이터 프레임(df2)에 있는 학생들의 점수들만 반환
# 점수가 없는 과목은 NaN을 반환
df1.merge(df2, how = 'right', on = 'student_id')


# 

# ### 2절. 데이터 요약

# #### 1. 그룹별 통계 요약

# In[9]:


# .groupby()

# 데이터프레임 생성
import pandas as pd
obj = {'student_id' : ['s1', 's2', 's3', 's4', 's5', 's6'],
        'stat_score' : [55, 90, 85, 71, 63, 99],
        'math_score' : [65, 99, 87, 75, 57, 88],
        'sex' : ['Female', 'Male', 'Female', 'Female', 'Male', 'Male'],
        'pre_level' : ['B', 'A', 'B', 'B', 'C', 'A']
       }
df = pd.DataFrame(obj)
print(df)


# In[10]:


# 그룹화한 후 데이터프레임 메소드 mean()을 통해 그룹화 평균 계산 가능
df.groupby('sex').mean()


# In[13]:


# 옵션 as_index = False
# 행인덱스로 행위치 인덱스 번호를 사용함
df.groupby('sex', as_index = False).mean()


# In[11]:


# 여러 열을 기준으로 그룹화하는 방법은 컬럼명을 담은 리스트를 사용
df.groupby(['sex', 'pre_level'], as_index = False).mean()


# In[12]:


# describe() 메소드는 여러 종류의 특정한 통계량들을 제공
df.groupby('sex', as_index = False).describe()


# In[13]:


# .agg
# 동시에 원하는 여러 통계량
import numpy as np
df.groupby('sex').agg([np.mean, sum])


# In[14]:


# 컬럼별로 상이한 통계량
df.groupby('sex', as_index = False).agg({'stat_score' : np.mean,'math_score' : sum})


# In[15]:


df.groupby('sex', as_index = False).agg({'stat_score' : [np.mean, np.median],
                                         'math_score' : [sum, max]})


# #### 2. 데이터에 함수 적용하기

# ##### 가. 시리즈에 함수 적용하기

# In[16]:


import pandas as pd

# 데이터프레임 생성
obj = {'student_id' : ['s1', 's2', 's3', 's4', 's5', 's6'],
        'stat_score' : [55, 90, 85, 71, 63, 99],
        'math_score' : [65, 99, 87, 75, 57, 88],
        'sex' : ['Female', 'Male', 'Female', 'Female', 'Male', 'Male'],
        'pre_level' : ['B', 'A', 'B', 'B', 'C', 'A']
       }
df = pd.DataFrame(obj)
print(df)


# In[17]:


# pre_level 컬럼 라벨인코딩
df['pre_level'].map({'A' : 0, 'B' : 1, 'C' : 2})


# In[18]:


# sex 컬럼의 요소를 한글로 변경
df['sex'].map({'Female' : '여자', 'Male' : '남자'})


# In[19]:


# 시리즈에 사용자 정의 함수 적용
def f(x) :
    return x ** 2 + 2*x - 5000  # 각 요소에 적용할 함수

df['stat_score'].map(f)


# ##### 나. 데이터프레임에 함수 적용하기

# In[20]:


# .apply
df[['stat_score', 'math_score']].apply(np.sum)


# In[21]:


df[['stat_score', 'math_score']].apply(np.sum, axis = 1)


# In[22]:


# apply 메소드 없이도 가능
df[['stat_score', 'math_score']].sum()


# In[23]:


df[['stat_score', 'math_score']].sum(axis = 1)


# -----

# ## 5장. 결측치와 이상치

# ### 1절. 결측치

# #### 1. 결측치 인식

# In[24]:


# 결측치가 포함되도록 데이터프레임 생성
obj = {'student_id' : ['s1', 's2', 's3', 's4', 's5', 's6'],
        'stat_score' : [None, 90, 85, 71, 63, None],
        'math_score' : [65, None, 87, 75, 57, 88],
        'sex' : ['Female', 'Male', 'Female', None, 'Male', 'Male'],
        'pre_level' : ['B', 'A', 'B', 'B', 'C', None]
       }
df = pd.DataFrame(obj)
df 


# In[25]:


# .isnull() : 결측치 여부 확인
df.isnull()


# In[26]:


# .isna() : 결측치 여부 확인
df.isna()


# In[27]:


# 컬럼별 결측치 개수
df.isnull().sum()


# In[28]:


# 행별 결측치 개수
df.isnull().sum(axis = 1)


# #### 2. 결측치 처리

# ##### 가. 결측치 제거

# In[29]:


# .dropna()
# 결측치가 있는 행을 모두 제거
df.dropna()


# In[30]:


# 열도 가능
df.dropna(axis = 1)


# In[31]:


# 'stat_score' 컬럼(시리즈)에서 결측인 요소 제거
df['stat_score'].dropna()


# In[32]:


# 'stat_score'과 'math_score'컬럼 중
# 결측치가 있는 모든 행을 제거
df[['stat_score', 'math_score']].dropna()


# ##### 나. 평균 대치법

# In[33]:


# 숫자형 컬럼만 추출
df1 = df[['stat_score','math_score']].copy() # copy() 원본 소실 방지
df1


# In[34]:


# .fillna
# 결측치를 모두 0으로 대치
df1.fillna(0)


# In[35]:


# 평균 대치
# 컬럼별 평균으로 대치
df1.fillna(df1.mean())


# ### 2절. 이상치

# #### 1. 이상치란

# 코드 없음

# #### 2. 이상치 인식과 제거

# ##### 가. 사분위수를 이용한 이상치 판별

# 코드 없음

# ##### 나. 이상치 제거

# In[36]:


# 데이터프레임 생성
obj = {'student_id' : ['s1', 's2', 's3', 's4', 's5', 's6'],
        'stat_score' : [55, 90, 85, 10, 88, 99],
        'math_score' : [65, 99, 67, 70, 57, 80],
       }
df = pd.DataFrame(obj)
print(df)


# In[37]:


# 'stat_score' 컬럼에 대하여 이상치를 판별하고 제거
q1 = df['stat_score'].quantile(0.25) # 제1사분위수
q3 = df['stat_score'].quantile(0.75) # 제3사분위수

iqr = q3 - q1 # IQR


# In[38]:


lower = q1 - 1.5 * iqr # lower whisker
upper = q3 + 1.5 * iqr # upper whisker


# In[39]:


# 이상치 판별
# upper whisker보다 크거나 lower whisker보다 작은 값들
df[(df['stat_score'] < lower) | (df['stat_score'] > upper)]


# In[40]:


# 이상치 제거
df[(df['stat_score'] > lower) & (df['stat_score'] < upper)]


# --- 

# ## 6장. 기타 데이터 전처리

# ### 1절. 텍스트 데이터 다루기

# #### 1. 문자열 데이터 전처리

# In[41]:


import pandas as pd

# 데이터프레임 불러오기
# 인코딩 옵션(encoding = 'CP949')을 사용해 한글이 깨지는 것 방지
df = pd.read_csv('data/예제/University_text_after.csv', encoding = 'CP949')
df


# In[42]:


# 시리즈객체.str[]
# 단순하게 인덱싱과 슬라이싱 가능
df['주소'].str[0:5] # 처음부터 5개의 문자열을 추출


# In[43]:


# str.startswith(pat)
# pat(문자열)로 시작한다면 True, 아니면 Fasle 반환
df['주소'].str.startswith("서울") # '서울'로 시작하면 True, 아니면 False


# In[44]:


# 주소가 '서울'로 시작하는 데이터셋 추출
df[df['주소'].str.startswith("서울")]


# In[45]:


# 주소가 '서울'로 시작하지 않는 데이터셋 추출
df[~df['주소'].str.startswith("서울")] # ~는 부정


# In[46]:


# str.endswith(pat)
# pat(문자열)로 끝난다면 True, 아니면 Fasle 반환
df['대학이름'].str.endswith("대학교") # '대학교'로 끝나면 True, 아니면 False


# In[47]:


# 대학이름이 '대학교'로 끝나는 데이터셋 추출
df[df['대학이름'].str.endswith("대학교")]


# In[48]:


# 대학이름이 '대학교'로 끝나지 않는 데이터셋 추출
df[~df['대학이름'].str.endswith("대학교")]


# In[49]:


# str.contains(pat)
# pat(문자열 또는 정규식)이 포함되면 True, 아니면 Fasle 반환
df['주소'].str.contains('광역시')


# In[50]:


# 주소에 '광역시'를 포함하는 데이터셋 추출
df[df['주소'].str.contains('광역시')]


# In[51]:


# str.split(pat=None)
# pat를 기준으로 문자열을 분할
df['주소'].str.split() # 공백을 기준으로 분리


# In[52]:


# str.find(sub, start=0, end=None)
# sub가 있는 위치를 start에서부터 end까지에서 찾아 위치값을 반환
df['주소'].str.find('구') # '구'가 있는 위치 인덱스


# In[53]:


# str.rfind(sub, start=0, end=None)
# sub가 있는 위치를 start에서부터 end까지에서 찾아 가장 높은 위치값을 반환
df['주소'].str.rfind(sub = ' ') # 가장 뒤에 있는 공백 위치


# In[54]:


# str.findall(pat)
# 모든 pat(문자열 또는 정규식) 항목을 반환
df['주소'].str.findall('\w+구') # 구로 끝나는 단어들을 모두 반환


# In[55]:


# str.replace(pat, repl)
# pat를 repl으로 대치함
df['주소'].str.replace(" ", "_") # 공백을 '_'로 대치함


# In[56]:


# 데이터프레임 불러오기
df = pd.read_csv('data/예제/University_text_before.csv', encoding = 'CP949')
df


# In[57]:


# str.strip(to_strip = None)
# 선행 및 후행 문자를 제거
df['주소'].str.strip(to_strip = '_^!#? /%')


# In[58]:


# str.lstrip(to_strip = None)
# 선행 문자를 제거
df['주소'].str.lstrip(to_strip = '_^!#? /%')


# In[59]:


# str.rstrip(to_strip=None)
# 후행 문자를 제거
df['주소'].str.rstrip(to_strip = '_^!#? /%')


# In[60]:


# 대소문자 변경
# .str.lower() : 모두 소문자로 변경
# .str.upper() : 모두 대문자로 변경
# .str.swapcase() : 소문자는 대문자, 대문자는 소문자로 변경 


# 

# ### 2절. 날짜 데이터 다루기

# #### 1. datetime형

# In[61]:


import pandas as pd

# 데이터프레임 불러오기
df = pd.read_csv('data/예제/University_date.csv', encoding = 'CP949')


# In[62]:


# 날짜 형태인 문자열
df['창립일']


# In[63]:


# pandas.to_datetime()
# 문자열 -> datetime형
pd.to_datetime(df['창립일']) # 구문 분석 자동


# In[64]:


# 문자열 -> datetime형
pd.to_datetime(df['창립일'], format = '%Y-%m-%d') # '네자리연도-월-일'로 구문 분석


# In[65]:


# 기존의 창립일 열을 datatime형을 변환
df['창립일'] = pd.to_datetime(df['창립일'])


# In[66]:


# 시계열생성
# 2000년 1월 1일부터 2000년 1월 10일까지 1일 단위로 생성
pd.date_range("2000-01-01", "2000-01-10")


# In[67]:


# 2000년 1월 1일부터 1일씩 증가(3번 반복)
pd.date_range("2000-01-01", periods = 3)


# In[68]:


# 2000년 1월부터 1달씩 증가(5번 반복)
# 각 달의 마지막 날짜가 반환됨(freq = "m")
pd.date_range("2000-01-01", periods = 5, freq = "m")


# ##### 2. 시계열 데이터 전처리

# In[69]:


# 데이터프레임 불러와 기존의 창립일 열을 datatime형을 변환
df = pd.read_csv('data/예제/University_date.csv', encoding = 'CP949')
df['창립일'] = pd.to_datetime(df['창립일'])

# 메소드 dt의 하위 메소드를 통한 날짜와 시간에 대한 정보를 확인
# dt.date : 날짜정보(연월일)
df['창립일'].dt.date


# In[70]:


# dt.year : 연도
df['창립일'].dt.year


# In[71]:


# dt.month : 월
df['창립일'].dt.month


# In[72]:


# dt.month_name() : 월이름
df['창립일'].dt.month_name()


# In[73]:


# dt.day : 일
df['창립일'].dt.day


# In[74]:


# dt.weekday : 요일번호
df['창립일'].dt.weekday


# In[75]:


# dt.day_name() : 요일이름
df['창립일'].dt.day_name()


# In[76]:


# dt.quarter : 분기
df['창립일'].dt.quarter


# In[77]:


# weekofyear : 연도기준 주
df['창립일'].dt.weekofyear


# In[78]:


# dayofyear : 연도기준 일
df['창립일'].dt.dayofyear


# In[79]:


# daysinmonth : 해당 월의 총 일수
df['창립일'].dt.daysinmonth


# In[80]:


# 2022-06-01 16:30:05부터 1초씩 증가(10번 반복)하는 시계열 생성
sr  = pd.Series(pd.date_range("2022-06-01 16:30:05", periods  = 5, freq = "s"))
sr


# In[81]:


# dt.time : 시간정보(시분초)
sr.dt.time


# In[82]:


# dt.hour : 시
sr.dt.hour


# In[83]:


# dt.minute : 분
sr.dt.minute


# In[84]:


# dt.second : 초
sr.dt.second


# # (끝)
