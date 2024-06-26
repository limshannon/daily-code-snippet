#!/usr/bin/env python
# coding: utf-8

# ## Secret PART 02. 작업형03 모의고사 2회차

# 01. 제공된 데이터(survey_subset.csv)는 성별에 따른 A 제품의 인지도 조사를 위해 남, 녀 각각 500명을 임의로 추출하여 실시된 응답 현황 데이터이다. 성별에 따라 1번 문항에 대한 응답이 차이가 있는 지를 동질성 검정(Test of Homogeneity)를 통해 답하시오. 가설은 아래와 같다.

# In[2]:


import pandas as pd
samp = pd.read_csv('survey_subset.csv', encoding = 'cp949')

# 필요한 패키지, 함수 호출
from scipy.stats import chi2_contingency

# 카이제곱 검정 수행
tb = pd.crosstab(samp['성별'], samp['1번문항'])
stat, pval, df, expected = chi2_contingency(tb)

# (a) 보통이다 응답 기대도수
E13 = expected[0,2].astype('int')
print(E13)

# (b) 검정통계량
stat = stat.astype('int')
print(stat)

# (c) p값/가설 검정 결과
pval = round(pval, 4)
print(pval)
print('기각')


# 02. 주어진 데이터(Cars93.csv)를 통해 미국 본토 회사(Origin) 여부에 따라 평균 프리미엄 자동차 가격(Max_Price)이 더 큰 지 t검정을 통해 답하고자 한다. 여기서, 두 그룹의 모분산은 동일하지 않다고 알려져있다. 가설은 아래와 같다.

# In[5]:


import pandas as pd
samp = pd.read_csv('Cars93.csv', encoding = 'cp949')

# 필요한 패키지, 함수 호출
from scipy.stats import ttest_ind

# 필요한 컬럼 각각 할당
Price = samp['Max_Price']
Origin = samp['Origin']

# 미국 본토 회사 프리미엄 자동차 가격
Price_USA = Price[Origin == 'USA'].reset_index(drop = True)

# 미국 본토가 아닌 회사 프리미엄 자동차 가격
Price_non = Price[Origin == 'non-USA'].reset_index(drop = True)

# (a) 미국 본토가 아닌 회사 프리미엄 자동차 가격의 표본 평균에서 미국 본토인 경우를 뺀 값
diff = Price_non.mean() - Price_USA.mean()
est = round(diff, 2)
print(est)

# 대응표본 t검정 수행
stat, pval = ttest_ind(Price_non, Price_USA, equal_var = False, alternative = 'less')

# (b) 검정통계량 = 점추정량/표준오차임을 이용
se = est/stat
se = round(se, 2)
print(se)

# (c) 검정통계량/기각 여부
stat = round(stat, 4)
print(stat)
print('채택')

