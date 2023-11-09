#!/usr/bin/env python
# coding: utf-8

# ## Secret PART 01. 작업형03 예상문제

# 01. 주어진 데이터(Rabbit_Five.csv)는 신약 개발을 위해 실험에 사용된 데이터이다. 실험군(MDL)과 대조군(Control) 간 혈압 변화(BPchange)가 차이가 있는 지를 쌍체표본 t-검정(paired t-test)를 통해 답하고자 한다.

# In[2]:


import pandas as pd
samp = pd.read_csv('Rabbit_Five.csv', encoding = 'cp949')

# 필요한 패키지, 함수 호출
from scipy.stats import ttest_rel

# 필요한 컬럼 각각 할당
BP_change = samp['BP_change']
Treatment = samp['Treatment']

# Treatment가 Control인 경우(대조군)와 MDL인 경우(실험군)의 BP_change 값 각각 할당
BPC_Treat = BP_change[Treatment == "MDL"].reset_index(drop = True)
BPC_Control = BP_change[Treatment == "Control"].reset_index(drop = True)

## (a) 점추정량 = mean(PC_Treat - PC_Control)
diff_avg = (BPC_Treat - BPC_Control).mean()
diff_avg = round(diff_avg, 2)
print(diff_avg)

## (b)-(c)

# 대응표본 t검정 수행
a = ttest_rel(BPC_Treat, BPC_Control)

# (b) 검정통계량
stat = a.statistic
stat = round(stat, 2)
print(stat)

# (c) p-값/기각여부
pval = a.pvalue
pval = round(pval, 3)
print(pval)
print('기각')

# 아래 코드처럼 바로 할당해서 해도 됨!
# stat, pval = ttest_rel(BPC_Treat, BPC_Control)


# 02. 주어진 데이터(mtcars2.csv)를 통해 변속기 종류(am)에 따라 마력(hp)에 대한 분산이 차이가 있는 지를 분산비 검정(F test to compare two variances)를 통해 답하고자 한다. 가설은 아래와 같다.

# In[3]:


import pandas as pd
samp = pd.read_csv('mtcars2.csv', encoding = 'cp949')

# 필요한 패키지, 함수 호출
from scipy.stats import f

# 필요한 컬럼 각각 할당
am = samp['am']
hp = samp['hp']

# 수동변속기(am=1)인 자동차의 마력(hp), 자동변속기(am=0)인 자동차의 마력(hp) 각각 할당
hp_manual = hp[am ==1].reset_index(drop = True)
hp_auto = hp[am == 0].reset_index(drop = True)

## (a) 수동변속 표본분산/자동변속 표본분산
var_ratio = hp_manual.var()/hp_auto.var()

## (b)-(c)

# F검정 수행
# (b) 검정통계량 = (a)와 같음
stat = var_ratio

# (c) 유의확률
# 자유도
df1, df2 = len(hp_manual) - 1, len(hp_auto) - 1

# F 분포로 확률 계산
pval = 1 - f.cdf(stat, dfn = df1, dfd = df2) # Pr[F > stat]
 
# 정답 출력을 위해 반올림
a = round(var_ratio, 2)
b = round(stat, 2)
c = round(pval, 3)

# 정답 출력
print(a) # (a)
print(b) # (b) 검정통계량
print(c) # (c) p-값
print('기각') # 기각여부


# 03. 주어진 데이터(고객_등급리스트.csv)를 통해 고객군(Segment)과 지역(Region)간 관련이 있는 지를 독립성 검정(Test of independence)을 통해 답하고자 한다. 가설은 아래와 같다.

# In[4]:


import pandas as pd
samp = pd.read_csv('고객_등급리스트.csv', encoding = 'cp949')

# 필요한 패키지, 함수 호출
from scipy.stats import chi2_contingency
import numpy as np

# 교차표 생성
tb = pd.crosstab(samp['Segment'], samp['Region'])

# 카이제곱 검정 수행
# chi2_contingency의 결과는 카이제곱통계량, 유의확률, 자유도, 기대도수를 반환함
chi2, pval, df, expected = chi2_contingency(tb)

# (a)E23 : expected의 (1,2) 인덱스 번호 추출
e23 = expected[1,2]
e23 = round(e23, 2)
print(e23)

# (b) 검정통계량
chi2 =  chi2.astype('int') # 정수 변환
print(chi2)

# (c) p값/기각 여부
pval = round(pval, 3)
print(pval)
print('채택')


# 04. 주어진 데이터(Cars93.csv)를 통해 가격(Price)이 정규 분포를 따르는 지를 샤피로-윌크 검정(Shapiro Wilk Test)를 통해 답하고자 한다. 가설은 아래와 같다.(단, 결측치는 무시할 것)

# In[5]:


import pandas as pd
samp = pd.read_csv('Cars93.csv', encoding = 'cp949')

# 필요한 패키지, 함수 호출
from scipy.stats import shapiro

# 필요한 컬럼 할당
Price = samp['Price'].copy().dropna()

# (a) 표본평균
avg = Price.mean()
avg = round(avg, 2)
print(avg)

# 샤피로 윌크 검정 수행
stat, pval = shapiro(Price)

# (b) 검정통계량
stat = round(stat, 2)
print(stat)

# (c) p값/기각 여부
pval = round(pval, 4)
pval = int(pval)
print(pval)
print('기각')


# 5. 주어진 데이터(Cars93.csv)를 통해  마일당엔진회전수(Rev_per_mile)과 마력(Horsepower)과의 상관관계를 알아보고 상관계수의 유의성을 피어슨 상관계수 검정(Pearson‘s Correlation Coefficient Test)를 통해 답하고자 한다. 가설은 아래와 같다.

# In[6]:


import pandas as pd
import numpy as np

samp = pd.read_csv('Cars93.csv')

# 상관분석에 필요한 컬럼명 저장
Horsepower = samp['Horsepower']
RPM = samp['Rev_per_mile']

# 필요한 패키지, 함수 호출
from scipy.stats import pearsonr

# 상관계수 검정 수행
rho, pval = pearsonr(Horsepower, RPM)

# (a) 표본상관계수
rho = round(rho, 3)
print(rho)

# (b) 검정통계량
stat = rho/np.sqrt( (1-rho**2)/(len(Horsepower) - 2) )
stat = round(stat, 2)
print(stat)

# (c) p값/기각 여부
pval = int(pval)
print(pval)
print('기각')


# 06. 제공된 데이터(USArrests.csv)는 미국 50개 주의 범죄와 체포와 관련된 데이터이다. 차원 축소를 주성분 분석을 통해 수행하고자 한다.

# In[7]:


import pandas as pd
samp = pd.read_csv('USArrests.csv')

# 필요한 패키지, 클래스 호출
from sklearn.decomposition import PCA

# PCA 수행
pca = PCA(n_components = 4) # 주성분객체 생성
pca.fit_transform(samp) 

# (a) 첫번째 주성분의 폭력범죄 기여 가중치
# pca.components_.T에서 행은 기존 컬럼(Merder, Assault, UrbanPop, Rape)
# 열은 1~4 주성분임
weight = pca.components_.T[1, 0]
weight = round(weight, 3)
print(weight)

# (b) 34번째 도시의 1주성분의 주성분 점수
score = pca.fit_transform(samp)[33,0]
score = round(score, 3)
print(score)

# (c)
# 주성분별 설명되는 분산 비율을 시리즈 객체로 저장
var_ratio = pd.Series(pca.explained_variance_ratio_)
result = round(var_ratio[0], 2)
print(result)
print(3)


# 07. 주어진 데이터(Cars93.csv)를 통해 마일당엔진회전수(Rev_per_mile), 중량(Weight), 길이(Length), 엔진크기(EngineSize)를 입력하면 중간가격(Price)을 예측하는 다중 선형 회귀 분석(linear regression)을 하고자 한다. (단, 해당 컬럼들의 결측치는 제거함)

# In[8]:


# !pip install statsmodels==0.13.5 
# Renuwal 된 문제 풀이
import pandas as pd
samp = pd.read_csv('Cars93.csv')

# 필요한 패키지, 클래스 호출
import statsmodels.api as sm

# 회귀분석 수행
colnm = ['Price', 'Rev_per_mile', 'Weight','Length', 'EngineSize'] # 회귀 분석에 필요한 컬럼 별도 지정
samp = samp[colnm].dropna() # 결측치 제거

# y, X에 각각 할당
y = samp['Price']
X = samp[['Rev_per_mile', 'Weight','Length', 'EngineSize']]
X = sm.add_constant(X) # 절편항 적합을 위해 상수벡터 추가

# 모델 적합
model = sm.OLS(y,X) # OLS 객체 생성
result = model.fit() # fit 메소드를 통해 모형 적합

# (참고) 
# result.params # 회귀 계수만 추출
# result.tvalues # t통계량만 추출

result.summary() # 해당 코드를 통해 회귀분석 통합 결과를 확인하고 값을 입력하면 됨

# 결과 출력
# (a) 결정계수
r_square = 0.396
print(r_square)

# (b) 문제의 의도는
# Weight의 추정 회귀 계수를 출력하는 것이다.
b = 0.0023
print(b)

# (c) 문제의 의도는
# Weight의 P>|t|을 통해 회귀 계수를 검정하는 것이다.
pval = 0.158
print(pval)

# (d) 문제의 의도는
# Weight의 회귀 계수에 대한 95% 신뢰구간을 구하는 것이다. 아래의 함수 결과를 통해 상한 값을 입력하면 됨
# result.conf_int(alpha = 0.05, cols=None)
upper = 0.005406
upper = round(upper,4)
print(upper)


# 08. 제공된 데이터(job.csv)는 취업 현황 분석을 위해 사용된 데이터의 일부이다. 여러 특성(x1, x2, x3)를 통해 취업 성공(y) 여부를 예측하는 로지스틱 회귀 분석을 하고자 한다. (특히 x2 컬럼은 성별에 대한 정보로, 모형 적합시 남성(M)을 1로, 여성(F)로 인코딩하여 분석)

# In[9]:


import pandas as pd
samp = pd.read_csv('job.csv')

# 필요한 패키지, 클래스 호출
from sklearn.linear_model import LogisticRegression
import numpy as np

# x2 컬럼 : M -> 1, F -> 0
samp['x2'] = samp['x2'].map({'M' : 1, 'F' : 0})

# y, X에 각각 할당
y = samp['y']
X = samp[['x1', 'x2', 'x3']]

# 회귀모형 객체 생성 후 적합
model = LogisticRegression(penalty = 'none').fit(X, y)

# 결과 출력
# (a) 절편항 추정 회귀 계수
b0 = round(model.intercept_[0], 3)
print(b0)

# (b) 여성에 비해 남성의 성공에 대한 오즈가 몇 배인지를 구하려면
#     오즈비 = 남성의 성공 오즈/여성의 성공 오즈
#            = x2 컬럼이 성별이므로 exp(beta2)를 구하면 됨
odds_ratio = round(np.exp(model.coef_[0][1]), 3)
print(odds_ratio)

# (c) 9번 째 사람의 성공 예측 확률
y_prob = round(model.predict_proba(X)[8,1], 4)
print(y_prob)
print(0)


# 09. 주어진 데이터(영화_순위리스트.csv)를 통해 장르별 예산의 평균에 차이가 있는 지를 분산분석(ANOVA)를 수행하기 전 등분산 검정(Homogeneity of Variance)인 Bartlett's Test을 수행하고자 한다. 가설은 아래와 같다.

# In[11]:


import pandas as pd
samp = pd.read_csv('영화_순위리스트.csv', encoding = 'cp949')

# 필요한 패키지, 함수 호출
from scipy.stats import bartlett
import numpy as np

# 필요한 컬럼 각각 할당
genre = samp['장르']
budget = samp['예산']

# 장르별 예산 값 할당
budget_thriller = budget[genre == 'Thriller']
budget_comedy = budget[genre == 'Comedy']
budget_drama = budget[genre == 'Drama']
budget_action = budget[genre == 'Action']

## (a) 합동분산(pooled variancer)
# 집단별 표본 분산
var_i = [budget_thriller.var(), budget_comedy.var(), budget_drama.var(), budget_action.var()]

# 집단별 관측치 수
n_i = [len(budget_thriller), len(budget_comedy), len(budget_drama), len(budget_action)]

# 합동분산 계산
N = sum(n_i)
k = 4 # 집단의 수

log_sp2 = np.log(sum(np.subtract(n_i, 1) * var_i)/(N-k))
log_sp2 = round(log_sp2, 3)

print(log_sp2)

## (b)-(c)
# Bartlett Test 수행
stat, pval = bartlett(budget_thriller, budget_comedy, budget_drama, budget_action)

# (b) 검정통계량
stat = round(stat, 2)
print(stat)

# (c) p-값/기각여부
pval = round(pval, 4)
print(pval)
print('기각')

