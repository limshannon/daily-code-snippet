#!/usr/bin/env python
# coding: utf-8

# ## (작업형1) 연습문제

# #### 1. Cars93 데이터셋의 Wheelbase 컬럼에 대해서 평균 값에서 표준편차의 1.5배, 2배, 2.5배를 더하거나 뺀 값들의 구간 내의 데이터들의 평균을 각각 구한 후 원래의 데이터 평균에서 뺐을 때 차이들의 합을 출력하여라. (단, 소수점 다섯째 자리에서 반올림하여 표현할 것)

# In[1]:


#### 연습문제1 Solution
import pandas as pd
exam1 = pd.read_csv('data/연습문제/Cars93.csv')

# Wheelbase 따로 할당
Wheelbase = exam1['Wheelbase']

# Wheelbase의 평균
Wheelbase_avg = Wheelbase.mean()

# Wheelbase의 표준편차
Wheelbase_sd = Wheelbase.std()

##### Case1. 평균 값에서 표준편차를 1.5배를 더하거나 빼는 경우
# 구간의 하한(Low_1)과 상한(Upp_1) 계산
Low_1 = Wheelbase_avg - 1.5 * Wheelbase_sd
Upp_1 = Wheelbase_avg + 1.5 * Wheelbase_sd

# 구간 내 데이터들의 평균
Avg_1 = Wheelbase[(Wheelbase > Low_1) & (Wheelbase < Upp_1)].mean()

# 원래의 데이터 평균에서 뺌
Case1 = Wheelbase_avg - Avg_1

##### Case2. 평균 값에서 표준편차를 2배를 더하거나 빼는 경우
# 구간의 하한(Low_2)과 상한(Upp_2) 계산
Low_2 = Wheelbase_avg - 2 * Wheelbase_sd
Upp_2 = Wheelbase_avg + 2 * Wheelbase_sd

# 구간 내 데이터들의 평균
Avg_2 = Wheelbase[(Wheelbase > Low_2) & (Wheelbase < Upp_2)].mean()

# 원래의 데이터 평균에서 뺌
Case2 = Wheelbase_avg - Avg_2

##### Case3. 평균 값에서 표준편차를 2.5배를 더하거나 빼는 경우
# 구간의 하한(Low_3)과 상한(Upp_3) 계산
Low_3 = Wheelbase_avg - 2.5 * Wheelbase_sd
Upp_3 = Wheelbase_avg + 2.5 * Wheelbase_sd

# 구간 내 데이터들의 평균
Avg_3 = Wheelbase[(Wheelbase > Low_3) & (Wheelbase < Upp_3)].mean()

# 원래의 데이터 평균에서 뺌
Case3 = Wheelbase_avg - Avg_3

##### 결과를 result에 할당
result = round(Case1 + Case2 + Case3, 4)

##### 결과 출력
print(result)


# --- 

# #### 2.Cars93 데이터셋의 Length 컬럼에 대해서 순위를 부여한 후, 1위부터 30위까지 값들의 표준편차를 구하고 소수점 셋째 자리까지 반올림하여 나타내어라. (단, 동점은 동일한 순위를 부여하되 평균내어 등수를 산정하며 최솟값을 1위로 할 것)

# In[2]:


#### 연습문제2 Solution
import pandas as pd
exam2 = pd.read_csv('data/연습문제/Cars93.csv')


# Length의 순위
rank = exam2['Length'].rank(method = 'average')

# 1위~30위까지만 추출
sub = exam2['Length'][rank <= 30]

# sub의 표준편차
sub_sd = sub.std()

##### 결과를 result에 할당
result = round(sub_sd, 3)

##### 결과 출력
print(result)


# --- 

# #### 3. Cars93 데이터셋의 Max_Price 컬럼과 Min_Price 컬럼에 대해서 각각 정렬한 후 정렬된 순서에 따라 레코드별로 Max_Price와 Min_Price의 차이를 산출하고 차이값에 대해 표준편차를 구하여라. (단, Max_Price의 정렬은 내림차순, Min_Price의 정렬은 오름차순으로 하며, 출력시 표준편차는 소수점 넷째 자리에서 반올림하여 표현할 것)

# In[2]:


#### 연습문제3 Solution
import pandas as pd
exam3 = pd.read_csv('data/연습문제/Cars93.csv')

# 내림차순으로 정렬해 MaxPrice_sort에 할당
MaxPrice_sort = exam3['Max_Price'].sort_values(ascending = False, ignore_index = True)

# 오름차순으로 정렬해 MinPrice_sort에 할당
MinPrice_sort = exam3['Min_Price'].sort_values(ignore_index = True)

# 차이 계산
# 메소드 .sort_values()에 ignore_index = True을 하지 않을 경우
# 정렬과 무관하게 정렬 전의 인덱스가 같은 값들끼리 차이를 계산하게 됨
diff = MaxPrice_sort - MinPrice_sort

# 차이에 대한 표준편차
diff_sd = diff.std()

##### 결과를 result에 할당
result = round(diff_sd, 3)

##### 결과 출력
print(result)


# --- 

# #### 4. Cars93 데이터셋의 Weight 컬럼을 Min-Max 정규화로 변환한 후, 0.5보다 작은 값들의 분산과 0.5보다 큰 값들의 분산의 차이를 구하여라.(단, 차이는 큰 값에서 작은 값을 빼서 구하며, 소수점 넷째 자리에서 반올림하여 표현할 것)

# In[3]:


#### 연습문제4 Solution
import pandas as pd
exam4 = pd.read_csv('data/연습문제/Cars93.csv')

# Weight 컬럼 Min-Max 정규화로 변환
Weight = exam4['Weight']
Weight_std = (Weight - min(Weight))/(max(Weight) - min(Weight))

# 0.5보다 작은 Weight들의 분산
var_under = Weight_std[Weight_std < 0.5].var()

# 0.5보다 큰 Weight들의 분산
var_over = Weight_std[Weight_std > 0.5].var()

# 차이 계산
diff = abs(var_over - var_under)

##### 결과를 result에 할당
result = round(diff, 3)

##### 결과 출력
print(result)


# ---

# #### 5. Cars93 데이터셋을 이용하여 Manufacturer, Origin 컬럼의 유일값 조합의 수와 Manufacturer 컬럼의 앞 두글자만 추출한 결과와 Origin 컬럼과의 유일값 조합  수의 차이를 구하여라. (단, 원래 유일값 조합 수에서 추출 이후 수를 뺄 것)

# In[4]:


#### 연습문제5 Solution
import pandas as pd
exam5 = pd.read_csv('data/연습문제/Cars93.csv')

##### 원래 유일값 조합의 수
# .unique(): 시리즈의 유일값을 추출하는 메소드이다.
# .nunique()는 데이터프레임의 각 컬럼별 유일값 수를 계산하는 메소드이다.
# .drop_duplicates(): 데이터프레임의 여러 컬럼들의 조합에 대한 유일값을 추출하는 메소드
uniq_raw = exam5[['Manufacturer', 'Origin']].drop_duplicates()
num_uniq_raw = uniq_raw.shape[0]

##### Manufacturer 컬럼 앞 두 글자만 추출한 결과와 Origin과 유일값 조합의 수
# Manufacturer 컬럼 앞 두 글자 추출
exam5['sub_str'] = exam5['Manufacturer'].str[:2]

# 유일값 조합의 수
uniq_new = exam5[['sub_str', 'Origin']].drop_duplicates()
num_uniq_new = uniq_new.shape[0]

##### 결과를 result에 할당
result = num_uniq_raw - num_uniq_new

##### 결과 출력
print(result)


# ---

# #### 6. Cars93 데이터셋을 이용하여 컬럼 Type, Man_trans_avail에 대한 그룹별 RPM 레코드수와 RPM 합계, 중앙값을 모두 구한 후, 그룹별 중앙값에서 그룹별 합계에서 레코드 수를 나눈 값들을 빼서 나온 결과의 총 원소 합을 구하여라. (단, 출력시 소수점은 첫째 자리에서 반올림하여 표현할 것)

# In[5]:


#### 연습문제6 Solution
import pandas as pd
exam6 = pd.read_csv('data/연습문제/Cars93.csv')

##### 그룹별  RPM 레코드 수
count_RPM_gp = exam6.groupby(['Type', 'Man_trans_avail'])['RPM'].count()

##### 그룹별 RPM 합계
sum_RPM_gp = exam6.groupby(['Type', 'Man_trans_avail'])['RPM'].sum()

##### 그룹별 RPM 중앙값
median_RPM_gp = exam6.groupby(['Type', 'Man_trans_avail'])['RPM'].median()

##### 그룹별 중앙값 - (그룹별 합계/레코드 수)을 계산한 후 모든 원소 합
calcul = sum(median_RPM_gp - sum_RPM_gp/count_RPM_gp)

##### 결과를 result에 할당
result = round(calcul, 0)

##### 결과 출력
print(result)


# ---- 

# #### 연습문제7. Cars93 데이터셋을 이용하여 RPM 컬럼의 결측치를 평균으로 대체하고 RPM와 Wheelbase 컬럼을 각각 z-점수 표준화한 후 표준화된 Wheelbase에 상수 –36을 곱한 값과 표준화된 RPM 컬럼의 차이값을 구하고 표준편차를 산출하여라. (단, 소숫점 셋째 자리까지 반올림하여 표현할 것)

# In[52]:


#### 연습문제7 Solution
import pandas as pd
exam7 = pd.read_csv('data/연습문제/Cars93.csv')

##### RPM 컬럼 결측치 평균 대체
avg = exam7['RPM'].mean() # RPM 컬럼의 결측치를 제외한 평균
exam7['RPM'] = exam7['RPM'].fillna(avg)

##### RPM 컬럼 z-점수 표준화
RPM_std = (exam7['RPM'] - exam7['RPM'].mean())/exam7['RPM'].std()

##### Wheelbase 컬럼 z-점수 표준화
Wheelbase_std = (exam7['Wheelbase'] - exam7['Wheelbase'].mean())/exam7['Wheelbase'].std()

##### 표준화된 Wheelbase에 상수 –36을 곱한 값과 표준화된 RPM 변수의 차이값
diff = Wheelbase_std * (-36) - RPM_std

##### 차이값의 표준 편차
diff_sd = diff.std()

##### 결과를 result에 할당
result = round(diff_sd, 3)

##### 결과 출력
print(result)


# ----

# #### 연습문제8. Cars93 데이터셋을 이용하여 먼저, Price 컬럼의 결측치를 평균으로 대체하고 Max_Price 변수와 Min_Price의 평균보다 작은 레코드만을 추출해 산출된 Origin 그룹별 Price의 합계를 구하고 다음으로 Price 컬럼의 결측치를 중앙값으로 대체하고 Price 컬럼이 Min_Price 컬럼의 제 3사분위수보다 작은 레코드만을 추출해 산출된 Origin별 Price의 합계를 Origin 그룹별로 합한 후 큰 값을 출력하여라. (단, 소숫점 이하는 모두 절삭하여 정수로 표현할 것)

# In[3]:


#### 연습문제8 Solution
import pandas as pd
exam8 = pd.read_csv('data/연습문제/Cars93.csv')

# 결측치 대체를 같은 컬럼에 두 번해야하는 문제이므로 이에 데이터프레임을 따로 복사함
df_case1 = exam8.copy()
df_case2 = exam8.copy()

##### Case1
# Price 컬럼의 결측치를 평균으로 대체
avg = df_case1['Price'].mean() # Price 컬럼의 결측치를 제외한 평균
df_case1['Price'] = df_case1['Price'].fillna(avg)
# (참고) df_case1['Price'].fillna(avg, inplace = True)는 다시 할당 없이 바로 변경 가능

###### Price가 Max_Price와 Min_Price의 평균보다 작은 데이터프레임을 추출
# Max_Price와 Min_Price의 컬럼별 평균
avg_MaxMin = df_case1[['Max_Price', 'Min_Price']].mean(axis = 1)

# Price가 위의 평균보다 작은 데이터프레임
sub_df_case1 = df_case1[df_case1['Price'] < avg_MaxMin]

# Origin 그룹별 Price의 합계
sum_case1 = sub_df_case1.groupby('Origin')['Price'].sum()

##### Case2
# Price 변수의 결측치를 중앙값으로 대체
med = df_case2['Price'].median() # Price 컬럼의 결측치를 제외한 중앙값
df_case2['Price'] = df_case2['Price'].fillna(med)
# (참고) df_case1['Price'].fillna(med, inplace = True)는 다시 할당 없이 바로 변경 가능

###### Price가 Min_Price의 제 3사분위수보다 작은 데이터프레임을 추출
# Min_Price의 제 3사분위수
q3 = exam8['Min_Price'].quantile(.75)

# Price가 위의 제 3사분위수보다 작은 데이터프레임
sub_df_case2 = df_case2[df_case2['Price'] < q3]

# Origin 그룹별 Price의 합계
sum_case2 = sub_df_case2.groupby('Origin')['Price'].sum()

##### 두 결과를 합한 후 가장 큰 원소
max_value = max(sum_case1 + sum_case2)

##### 결과를 result에 할당
import numpy as np
result = int(np.floor(max_value)) # int(max_value)만 해도 됨

##### 결과 출력
print(result)


# ---- 

# #### 연습문제9. Cars93 데이터셋에서 ‘Price’ 컬럼은 ‘Min_Price’와 ‘Max_Price’의 평균으로 알려져있다. 이와 같은 사실을 통해 ‘Price’ 컬럼의 결측치의 원래의 값을 계산한 후, ‘Price’가 14.7보다 작거나 25.3보다 크면서 ‘Large’ 타입인 레코드 수를 계산하여라.

# In[8]:


#### 연습문제9 Solution
import pandas as pd
exam9 = pd.read_csv('data/연습문제/Cars93.csv')

##### 'Price' 컬럼의 결측치의 원래의 값을 계산
### 컬럼들 시리즈로 별도 저장
Price = exam9['Price'].copy()
Max_Price = exam9['Max_Price'].copy()
Min_Price = exam9['Min_Price'].copy()
Type = exam9['Type'].copy()

### 'Price' 컬럼이 결측인 조건
cond_na = Price.isna()

### 'Price'가 결측치인 경우만 'Min_Price'와 'Max_Price'의 평균을 할당
Price[cond_na] = (Max_Price[cond_na] + Min_Price[cond_na])/2

###### ‘Price’가 14.7보다 작거나 25.3보다 크면서 ‘Large’ 타입인 레코드 수
# 조건1
cond1 = Price < 14.7

# 조건2
cond2 = (Price > 25.3) & (Type == 'Large')

# 해당하는 조건
cond = cond1 | cond2

##### 결과를 result에 할당
result = exam9[cond].shape[0]

##### 결과 출력
print(result)


# ----

# #### 연습문제10. Cars93 데이터셋에서 ‘Make’ 컬럼을 이용하여 제조사가 ‘Chevrolet’, ‘Pontiac’, ‘Hyundai’이면서 ‘AirBags’이 ‘Driver’에만 있는 경우의 레코드 수를 계산하여라.

# In[9]:


#### 연습문제10 Solution
import pandas as pd
exam10 = pd.read_csv('data/연습문제/Cars93.csv')

#### 컬럼들 시리즈로 별도 저장
Make = exam10['Make'].copy()
AirBags = exam10['AirBags'].copy()

#### 제조사가 ‘Chevrolet’, ‘Pontiac’, ‘Hyundai’인 경우
### (위치 인덱스 기준) 12, 16, 72, 74번 문자열 앞에 공백이 포함되어있음
# Make[Make.str[0] == ' '] 확인 코드

### 선행 문자(공백) 제거
Make = Make.str.strip()

### 조건
# 튜플로 입력 시 여러 문자열로 시작하는 경우에 대한 부울 결과를 찾을 수 있음
# 문자열이 'Chevrolet' 또는 'Pontiac' 또는 'Hyundai'로 시작하면 True를 반환함
cond_1 = Make.str.startswith(('Chevrolet', 'Pontiac', 'Hyundai'))

#### 'AirBags'이 'Driver'에만 있는 경우
cond_2 = (AirBags == 'Driver only')

##### 결과를 result에 할당
result = sum(cond_1 & cond_2)

##### 결과 출력
print(result)


# # (끝)
