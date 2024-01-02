#!/usr/bin/env python
# coding: utf-8

# ## Secret PART 02. 작업형03 모의고사 1회차

# 01. 주어진 데이터(Cars93.csv)를 통해 자동차 타입이 Small인 경우보다 Large인 경우의 평균 가격에 차이가 있는 지 t검정을 통해 답하고자 한다. 가설은 아래와 같다.(단, 등분산인 정규 표본을 가정)

# In[1]:


import pandas as pd
samp = pd.read_csv('Cars93.csv', encoding = 'cp949')

# 필요한 패키지, 함수 호출
from scipy.stats import ttest_ind, t

# 필요한 컬럼 각각 할당
Type = samp['Type']
Price = samp['Price']

# 타입이 Small인 경우와 Large인 경우의 Price 값 각각 할당
Price_Large = Price[Type == "Large"].reset_index(drop = True).dropna()
Price_Small = Price[Type == "Small"].reset_index(drop = True).dropna()

# (a) 점추정량
est = (Price_Large.mean() - Price_Small.mean())
est = round(est, 2)
print(est)

# 대응표본 t검정 수행
stat, pval = ttest_ind(Price_Large, Price_Small, equal_var = True)

# (b) 검정통계량 = 점추정량/표준오차임을 이용
se = est/stat
se = round(se, 2)
print(se)

# (c) 95% 신뢰하한
# 자유도
df = len(Price_Large) + len(Price_Small) - 2

# 95% 신뢰하한
CI = t.interval(.95, df, loc = est, scale = se)[0] 

# 신뢰상한/가설 기각 여부
lower = round(CI, 1)
print(lower)
print('기각')


# 02. 주어진 데이터(dices.csv)는 주사위를 던졌을 때 1~6이 나올 확률이 1/6로 동일하다는 이론이 실제로는 다르다는 것을 보이기 위해서 주사위를 500번 던진 결과이다.  이를 적합도 검정을 통해 답하고자 한다. 가설은 아래와 같다.

# In[1]:


import pandas as pd
samp = pd.read_csv('dices.csv', encoding = 'cp949')

# 필요한 패키지, 함수 호출
from scipy.stats import chisquare

# 필요한 컬럼 각각 할당
tb = samp['scale'].value_counts().sort_index()

# 기대도수
expected = tb.sum()*(1/6)
expected = expected.astype('int') # 정수형 변환
print(expected)

# 적합도 검정 수행
stat, pval = chisquare(tb)

# 검정통계량
stat = int(stat)
print(stat)

# p값/기각 여부
pval = int(pval)
print(pval)
print('기각')

