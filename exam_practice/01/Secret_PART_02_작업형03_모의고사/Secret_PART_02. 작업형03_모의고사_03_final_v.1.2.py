#!/usr/bin/env python
# coding: utf-8

# ## Secret PART 02. 작업형03 모의고사 3회차

# 01. 주어진 데이터(영화_순위리스트.csv)를 통해 장르(genre)별 예산(budget)들의 평균에 차이가 있는 지를 분산분석(ANOVA)를 통해 답하고자 한다. 여기서, 앞의 01번의 결과와 관계 없이 모든 그룹의 모분산은 동일하다고 알려져있다. 가설은 아래와 같다.

# In[1]:


import pandas as pd
samp = pd.read_csv('영화_순위리스트.csv', encoding = 'cp949')

# 필요한 패키지, 함수 호출
from scipy.stats import f_oneway

# 필요한 컬럼 각각 할당
genre = samp['장르']
budget = samp['예산']

# 장르별 예산 값 할당
budget_thriller = budget[genre == 'Thriller']
budget_comedy = budget[genre == 'Comedy']
budget_drama = budget[genre == 'Drama']
budget_action = budget[genre == 'Action']

## (a)
# 집단별 표본 평균
avg_i = [budget_thriller.mean(), budget_comedy.mean(), budget_drama.mean(), budget_action.mean()]
a = max(avg_i) - min(avg_i) # 가장 평균 예산이 큰 장르의 표본평균에서 작은 장르의 표본평균을 뺀 값
a = round(a, 1)
print(a)

## (b)-(c)
stat, pval = f_oneway(budget_thriller, budget_comedy, budget_drama, budget_action)

# (b) 검정통계량
stat = round(stat, 2)
print(stat)

# (c) p-값/기각여부
pval = round(pval, 4)
print(pval)
print('채택')


# 02. 주어진 데이터(영화_순위리스트.csv)를 통해 장르(genre)별 트레일러뷰수의 평균에 차이가 있는 지를 검정하기 위해, 분산분석(ANOVA)을 수행한 결과, 귀무가설이 귀각되었다. 이에 따라 사후 검정을 수행하고자 한다. 가설은 아래와 같다.

# In[2]:


import pandas as pd
samp = pd.read_csv('영화_순위리스트.csv', encoding = 'cp949')

# 필요한 패키지, 함수 호출
from statsmodels.stats.multicomp import pairwise_tukeyhsd

posthoc = pairwise_tukeyhsd(samp["트레일러_뷰수"], samp["장르"], alpha=0.05)
print(posthoc)


# In[5]:


print(11836.686) # (a) 답
print(0) # (b) 답
print(-21075.666) # (c) 답
print(1) # (d) 답

