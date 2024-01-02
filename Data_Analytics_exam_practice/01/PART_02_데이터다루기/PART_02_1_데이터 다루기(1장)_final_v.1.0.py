#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NumPy 패키지 불러오기
import numpy as np # 넘파이 패키지를 np라는 별칭으로 호출


# ## PART 02) 데이터 다루기

# ## 1장. NumPy를 활용한 데이터 다루기

# ### 1절. NumPy의 배열객체

# #### 1. ndarray 배열

# 코드 없음

# #### 2. ndarray 특징

# 코드 없음

# #### 3. ndarray 생성

# In[2]:


# 1d-array 생성
# array(object) : 객체(object)를 배열로 반환함 
np.array([1, 3, 5, 7, 9]) # 리스트객체를 1차원 배열로


# In[3]:


np.array('dataedu') # 문자열객체를 1차원 배열로


# In[4]:


# arange([start, ]stop, [step, ])
# start 숫자에서 step 간격으로 증가해 end 전(end를 포함하지 않는) 숫자까지 숫자열 생성
np.arange(7) # 0에서 6까지 정수열 1차원 배열 생성


# In[6]:


# arange는 정수형만 가능한 range()와는 달리 실수도 가능
np.arange(1, 6, 0.5) # 1에서 5까지 0.5씩 증가하는 실수열 1차원 배열 생성


# In[7]:


# (참고) range()는 step에 실수가 되지 않아 오류
range(1, 6, 0.5)


# In[8]:


# ones(shape)
# shape 값에 따른 모든 값이 1.(실수)인 배열을 생성
np.ones(5)  # 모든 값이 1.이고 요소가 5개인 1차원 배열 생성


# In[9]:


# zeros(shape)
# shape 값에 따른 모든 값이 0.(실수)인 배열을 생성
np.zeros(5) # 모든 값이 0.이고 요소가 5개인 1차원 배열 생성


# In[10]:


# full(shape, fill_value)
# shape 값에 따른 모든 값이 fill_value인 배열을 생성
np.full(5, 4.) # 모든 값이 4.(실수)이고 요소가 5개인 1차원 배열 생성


# In[11]:


# 배열 객체의 데이터 타입(dtype)
# 4가 정수이므로 배열은 정수형이 됨
np.full(5, 4).dtype


# In[12]:


# 4.은 실수이므로 배열은 실수형이 됨
np.full(5, 4.).dtype 


# In[13]:


# 4가 정수이지만, dtype인자를 통해 float로 지정하면 실수형이 됨
np.full(5, 4, dtype = 'float').dtype


# In[14]:


# 2d-array 생성
# 리스트 객체를 2차원 배열로
np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]) # 2행 5열


# In[15]:


# 리스트 객체를 2차원 배열로
np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 3행 3열


# In[16]:


# 모든 값이 1(정수)이고 행의 수가 2, 열의 수가 5개인 2차원 배열 생성
np.ones((2,5), dtype = 'int') # dtype 생략할 경우 실수로 자동 생성


# In[17]:


# 모든 값이 0(정수)이고 행의 수가 2, 열의 수가 5개인 2차원 배열 생성
np.zeros((2,5), dtype = 'int') # dtype 생략할 경우 실수로 자동 생성


# In[18]:


# 모든 값이 5(정수)이고 행의 수가 2, 열의 수가 5개인 2차원 배열 생성
np.full((2,5), 5)


# In[19]:


# 주 대각성분이 1인 행과 열의 수가 5개인 정사각 2차원 배열 생성(항등행렬과 유사)
np.identity(5)


# In[20]:


# 3d-array 생성
# 리스트 객체를 3차원 배열로
np.array([
    [[1, 2, 3, 4], [5, 6, 7, 8]], # 0층
    [[2, 4, 6, 8], [10, 12, 14, 16]], # 1층
    [[1, 3, 5, 7], [9, 11, 13, 15]] # 2층
         ]) # 3층 2행 4열


# In[21]:


# 모든 값이 1이고 3층 2행 4열인 3차원 배열 생성
np.ones((3, 2, 4), dtype = 'int') # dtype 생략할 경우 실수로 자동 생성


# In[22]:


# 모든 값이 0이고 3층 2행 4열인 3차원 배열 생성
np.zeros((3, 2, 4), dtype = 'int') # dtype 생략할 경우 실수로 자동 생성


# In[23]:


# 모든 값이 5이고 3층 2행 4열인 3차원 배열 생성
np.full((3, 2, 4), 5)


# #### 4. ndarray 정보 확인

# In[15]:


# ndarray 생성
# 1d-array
arr_1d = np.array([1, 2, 3, 4])

# 2d-array
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) # 2행 4열

# 3d-array
arr_3d = np.array([
    [[1, 2, 3, 4], [5, 6, 7, 8]], # 0층
    [[2, 4, 6, 8], [10, 12, 14, 16]], # 1층
    [[1, 3, 5, 7], [9, 11, 13, 15]] # 2층
    ]) # 3층 2행 4열


# In[16]:


# .shape
# 배열객체의 [층,] 행, [열]의 수
print(arr_1d.shape) # 1d-array의 경우 길이정보가 (길이, )와 같은 형태로 반환됨
print(arr_2d.shape)
print(arr_3d.shape) 


# In[17]:


# .size
# 배열객체의 총 원소의 수
print(arr_1d.size)
print(arr_2d.size)
print(arr_3d.size)


# In[18]:


# .ndim
# 배열객체의 차원의 수
print(arr_1d.ndim)
print(arr_2d.ndim)
print(arr_3d.ndim)


# 

# ### 2절. NumPy의 연산

# #### 1. ndarray 인덱싱과 슬라이싱

# ##### 가. 인덱싱

# In[19]:


# 인덱싱
# 1d-array객체 생성
arr_1d = np.array([1, 2, 3, 4, 5])


# In[20]:


print(arr_1d[0]) # 가장 처음(0번)으로부터 값 하나 참조
print(arr_1d[2]) # 2번 인덱스 위치로부터 값 하나 참조
print(arr_1d[4]) # 가장 마지막(4번)으로부터 값 하나 참조
print(arr_1d[-1]) # 음수로도 가능(-1번으로부터 값 하나 참조)


# In[21]:


# 2d-array객체 생성
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 7],
                   [6, 8, 9],
                   [11, 17, 19]]) # 4행 3열


# In[22]:


# Case1
# 처음(0번) 행/열 인덱스로부터 값 하나 참조
print(arr_2d[0,0])

# Case1(음수로 할 경우)
# –4번 행 인덱스와 –3번 열 인덱스로부터 값 하나 참조
print(arr_2d[-4,-3])

# Case2
# 2번 행 인덱스와 1번 열 인덱스로부터 값 하나 참조
print(arr_2d[2,1])

# Case2(음수로 할 경우)
# -2번 행/열 인덱스로부터 값 하나 참조
print(arr_2d[-2,-2])


# In[32]:


# 음수 인덱싱의 활용 예

# Case1
# 3번 행 인덱스와 2번 열 인덱스로부터 값 하나를 참조하는 것보다
# 마지막 행열에 위치하므로 -1번 행열 인덱스로부터 참조하는 것이 빠르다.
print(arr_2d[-1,-1])

# Case2
# 가장 처음 행(0번)과 가장 마지막 열(-1번) 인덱스로부터 값 하나 참조
print(arr_2d[0,-1])

# Case3
# 가장 마지막 행(-1번)과 가장 처음 열(0번) 인덱스로부터 값 하나 참조
print(arr_2d[-1,0])


# ##### 나. 슬라이싱

# In[23]:


# 슬라이싱
# 1d-array객체 생성
arr_1d = np.array([1, 2, 3, 4, 5])


# In[24]:


# Case1
# 처음(0번)부터 2번 인덱스 번호 사이의 구간 추출
print(arr_1d[0:2])
print(arr_1d[:2]) # k1 = 0으로 생략 가능


# In[25]:


# Case2
# 1번부터 4번 사이의 인덱스 번호 사이의 구간 추출
print(arr_1d[1:4])

# –4번부터 -1번 사이의 인덱스 번호 사이의 구간 추출
print(arr_1d[-4:-1])

# 1번부터 -1번 사이의 인덱스 번호 사이의 구간 추출
print(arr_1d[1:-1])


# In[26]:


# Case3
# 3번부터 마지막(5번) 사이의 인덱스 번호 사이의 구간 추출
print(arr_1d[3:5])

# 3번부터 마지막 사이의 인덱스 번호 사이의 구간 추출
print(arr_1d[3:]) # k2=길이정보로 생략 가능
 
# -2번부터 마지막 사이의 인덱스 번호 사이의 구간 추출
print(arr_1d[-2:])


# In[27]:


# 2d-array객체 생성
arr_2d = np.array([[1, 2, 3, 5],
                   [1, 3, 5, 7],
                   [2, 4, 6, 9],
                   [7, 11, 13, 17]]) # 4행 4열


# In[28]:


# Case1
# 행 : 전체
# 열 : 처음(0번)부터 1번 열 인덱스 사이의 구간
print(arr_2d[0:4, 0])
print(arr_2d[:4, 0]) # r1=0으로 생략 가능
print(arr_2d[:, 0]) # r1=0이고 r2=행길이므로 생략 가능


# In[29]:


# Case2
# 행/열 : 처음(인덱스 0번)부터 인덱스 3번(-1번) 사이의 구간
print(arr_2d[0:3, 0:3])
print(arr_2d[:3, :3]) # r1=c1=0으로 생략 가능
print(arr_2d[:-1, :-1]) # r1=c1=0으로 생략 가능


# In[30]:


# Case3
# 행/열 : 처음 바로 뒤(1번 또는 -3번)부터 마지막 바로 앞(-1번 또는 3번) 사이의 구간
print(arr_2d[1:3, 1:3])
print(arr_2d[-3:-1, -3:-1])
print(arr_2d[1:-1, 1:-1])


# In[31]:


# Case4
# 행 : 처음 바로 뒤(1번 또는 -3번)부터 마지막 바로 앞(-1번 또는 3번) 사이의 구간
# 열 : 전체
print(arr_2d[1:3, 0:4])
print(arr_2d[1:3, :]) # c1=0이고 c2=열길이이므로 생략 가능
print(arr_2d[1:-1, :]) # c1=0이고 c2=열길이이므로 생략 가능


# In[32]:


# Case5
# 행 : 마지막 행
# 열 : 열 전체
print(arr_2d[3, 0:4])
print(arr_2d[3, :]) # c1=0이고 c2=열길이이므로 생략 가능
print(arr_2d[-1, :]) # c1=0이고 c2=열길이이므로 생략 가능


# In[33]:


# Case6
# 행/열 : 2번(또는 -2번)부터 마지막 인덱스 사이의 구간
print(arr_2d[2:4, 2:4])
print(arr_2d[2:, 2:]) # c2=열길이이므로 생략 가능
print(arr_2d[-2:, -2:]) # c2=열길이이므로 생략 가능


# #### 2. ndarray 원소별 연산

# ##### 가. 단일 배열객체의 범용 함수

# In[34]:


# 단일 배열객체에 대한 범용 함수
# 1d-array객체 생성
arr = np.array([0, 1, -4, 9, -16, 25])
print(arr)


# In[35]:


# abs : 원소별 절대값을 반환(abs는 파이썬 기본 내장함수로도 가능함)
arr_abs = np.abs(arr)
print(arr_abs)

# fabs : 원소별 절대값을 반환(abs보다 빠름)
arr_fabs = np.fabs(arr)
print(arr_fabs)


# In[36]:


# sqrt : 원소별 제곱근 값 반환
# 음수는 nan 반환됨
arr_sqrt = np.sqrt(arr)
print(arr_sqrt)


# In[37]:


# square : 원소별 제곱 값을 반환
arr_square = np.square(arr)
print(arr_square)


# In[38]:


# exp : 원소별 밑이 e인 지수 함수 값을 반환
arr_exp = np.exp(arr)
print(arr_exp)


# In[39]:


# log : 원소별 자연로그 값을 반환
# 음수는 nan, 0은 -inf 반환됨
arr_log = np.log(arr)
print(arr_log)

# log10 : 원소별 상용로그 값을 반환
# 음수는 nan, 0은 -inf 반환됨
arr_log10 = np.log10(arr)
print(arr_log10)


# In[40]:


# sign : 원소별 부호 값을 반환
# 양수면 1, 음수면 -1, 0은 0을 반환
arr_sign = np.sign(arr)
print(arr_sign)


# In[41]:


# 단일 배열객체의 소숫점을 처리하는 범용 함수
# 1d-array객체 생성
arr = np.array([1.15, -2.33, 3.457, -4.095])
print(arr)


# In[42]:


# round : 원소별 소수점을 원하는 자릿수까지 반올림한 값을 반환
arr_round = np.round(arr) # default는 0
print(arr_round)

arr_round_decimals_1 = np.round(arr, 1) #decimals 인자 값만 입력해도 됨
print(arr_round_decimals_1)


# In[43]:


# ceil : 원소별 소수점을 올림한 값을 반환
arr_ceil = np.ceil(arr)
print(arr_ceil)


# In[44]:


# floor : 원소별 소수점을 내림한 값을 반환
arr_floor = np.floor(arr)
print(arr_floor)


# In[45]:


# trunc : 원소별 소수점을 잘라버린 값을 반환
arr_trunc = np.trunc(arr)
print(arr_trunc)


# ##### 나. 서로 다른 배열객체의 범용 함수

# In[46]:


# 서로 배열객체에 대한 범용 함수
# 1d-array객체 생성
arr_1 = np.arange(5) # 0 1 2 3 4
print(arr_1)

arr_2 = np.arange(1, 10, step = 2) # 1 3 5 7 9
print(arr_2)


# In[47]:


# add : 두 배열객체의 원소 별 덧셈
arr_add_1 = np.add(arr_1, arr_2)
print(arr_add_1)

# 연산자 +로 대신 가능
arr_add_2 = arr_1 + arr_2
print(arr_add_2)


# In[48]:


# subtract : 두 배열객체의 원소 별 뺄셈
arr_subtract_1 = np.subtract(arr_1, arr_2)
print(arr_subtract_1)

# 연산자 -로 대신 가능
arr_subtract_2 = arr_1 - arr_2
print(arr_subtract_2)


# In[49]:


# multiply : 두 배열객체의 원소 별 곱셈
arr_multiply_1 = np.multiply(arr_1, arr_2)
print(arr_multiply_1)

# 연산자 *로 대신 가능
arr_multiply_2 = arr_1 * arr_2
print(arr_multiply_2)


# In[50]:


# divide : 두 배열객체의 원소 별 나눗셈
arr_divide_1 = np.divide(arr_1, arr_2)
print(arr_divide_1)

# 연산자 /로 대신 가능
arr_divide_2 = arr_1 / arr_2
print(arr_divide_2)


# In[51]:


# mod : 두 배열객체의 원소 별 나눈 후 나머지
arr_mod_1 = np.mod(arr_1, arr_2)
print(arr_mod_1)

# 연산자 %로 대신 가능
arr_mod_2 = arr_1 % arr_2
print(arr_mod_2)


# In[52]:


# 두 배열객체가 서로 다른 길이, 차원, shape를 가질 경우
# Case1. 길이가 다른 경우
# 길이가 3인 1d-array 생성
arr_1d_3 = np.array([1, 2, 3])
print(arr_1d_3) # 확인
print(arr_1d_3.shape) # shape 확인

# 길이가 1인 1d-array 생성
arr_1d_1 = np.array([5])
print(arr_1d_1) # 확인
print(arr_1d_1.shape) # shape 확인

# 두 배열객체의 합은 길이가 3인 1d-array가 됨
arr_add = arr_1d_3 + arr_1d_1
print(arr_add) # 확인
print(arr_add.shape) # shape 확인


# In[53]:


# Case2. 차원이 다른 경우
# 행길이 3, 열길이 3인 2d-array 생성
arr_2d_3x3 = np.array([[1, 4, 7],
                       [2, 5, 8],
                       [3, 6, 9]])

print(arr_2d_3x3) # 확인
print(arr_2d_3x3.shape) # shape 확인

# 길이가 3인 1d-array 생성
arr_1d_3 = np.array([1, 0, -1])
print(arr_1d_3) # 확인
print(arr_1d_3.shape) # shape 확인

# 두 배열객체의 합은 행길이 3, 열길이 3인 2d-array가 됨
arr_add = arr_2d_3x3 + arr_1d_3
print(arr_add) # 확인
print(arr_add.shape) # shape 확인


# In[54]:


# Case3. shape가 다른 경우
# 행길이 1, 열길이 3인 2d-array 생성
arr_2d_1x3 = np.array([[1, 2, 3]])

print(arr_2d_1x3) # 확인
print(arr_2d_1x3.shape) # shape 확인
print(arr_2d_1x3.ndim) # 차원 확인

# 행길이 3, 열길이 1인 2d-array 생성
arr_2d_3x1 = np.array([[1],
                       [0],
                       [-1]]) 

print(arr_2d_3x1) # 확인
print(arr_2d_3x1.shape) # shape 확인
print(arr_2d_3x1.ndim) # 차원 확인

# 두 배열객체의 합은 행길이 3, 열길이 3인 2d-array가 됨
arr_add = arr_2d_1x3 + arr_2d_3x1
print(arr_add) # 확인
print(arr_add.shape) # shape 확인


# 

# ### 3절. NumPy의 주요 함수와 메소드

# #### 1. 형상 변환 메소드

# In[55]:


# .reshape()
# 배열 객체를 입력된 shape로 변환

# 길이가 8인 1d-array객체 생성
arr_1d = np.arange(8)
print(arr_1d)

# 행 길이가 2이고 열 길이가 4인 2d-array객체로 변환
arr_2d_2x4 = arr_1d.reshape(2,4)
print(arr_2d_2x4)

# 행 길이가 4이고 열 길이가 2인 2d-array객체로 변환
arr_2d_4x2 = arr_1d.reshape(4,2)
print(arr_2d_4x2)

# 행 길이가 3이고 열 길이가 3인 2d-array객체로 변환
arr_2d_3x3 = arr_1d.reshape(3,3)


# In[56]:


# reshape 메소드 활용 예 1 : (3,3)인 2차원 배열 변환
# 길이가 9인 1d-array객체 생성
arr_1d = np.arange(9)
print(arr_1d)

# 행 길이가 3이고 열 길이가 3인 2d-array객체로 변환
arr_2d_3x3 = arr_1d.reshape(3,3)
print(arr_2d_3x3)

# -1 이용해 2d-array객체로 변환
# 행의 수를 3으로 지정하고 남은 차원(열)은 알아서 정함
arr_2d_3xN = arr_1d.reshape(3,-1)
print(arr_2d_3xN)

# 열의 수를 3으로 지정하고 남은 차원(행)은 알아서 정함
arr_2d_Nx3 = arr_1d.reshape(-1,3)
print(arr_2d_Nx3)


# In[57]:


# reshape 메소드 활용 예 2 : (3,3)인 2차원 배열 생성
# 원소가 0~8이면서 (3,3)인 2d-array객체 생성
# 방법1
arr1 = np.array([[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]])
print(arr1)

# 방법2
arr2 = np.arange(9).reshape(3,3)
print(arr2)


# In[58]:


# reshape 메소드 활용 예 3 : (3,2,2)인 3차원 배열 생성
# 원소가 0~11이면서 층이 3이고 행/열 길이가 2인 3d-array객체 생성
# 방법1
arr4 = np.arange(12).reshape(3,2,2)
print(arr4)

# 방법2
arr5 = np.arange(12).reshape(3,2,-1) # 층 수 3, 행 수 2로 지정, 남은 차원(열) 자동
print(arr5)

# 방법3
arr6 = np.arange(12).reshape(-1,2,2) # 행/열의 수는 2로 지정, 남은 차원(층) 자동
print(arr6)


# In[59]:


# 다차원 배열 객체를 1차원 배열로 변환(평탄화)

# reshape 메소드를 활용
# (3,3)인 2차원 배열을 길이가 9인 1차원 배열로 변환
arr1_1d = arr1.reshape(9)
print(arr1_1d)

# (3,2,2)인 3차원 배열을 길이가 12인 1차원 배열로 변환
arr4_1d = arr4.reshape(12)
print(arr4_1d)


# In[60]:


# .ravel()
# 다차원 배열 객체를 1차원으로 평탄화함

arr2_1d = arr2.ravel() # 길이를 입력하지 않아도 됨
print(arr2_1d)

arr5_1d = arr5.ravel() # 길이를 입력하지 않아도 됨
print(arr5_1d)


# In[61]:


# .flatten()
# 다차원 배열 객체를 1차원으로 평탄화함
arr4_1d = arr4.flatten() # 길이를 입력하지 않아도 됨
print(arr4_1d)

arr6_1d = arr6.flatten() # 길이를 입력하지 않아도 됨
print(arr6_1d)


# In[62]:


# .transpose()
# 배열의 축을 교환함
arr_2d = np.arange(8).reshape(4,2)
print(arr_2d)

arr_2d_T = arr_2d.transpose() # 축 교환
print(arr_2d_T)

# .T 로도 바로 가능
print(arr_2d.T)


# #### 2. 통계 함수

# 코드 없음

# ----
