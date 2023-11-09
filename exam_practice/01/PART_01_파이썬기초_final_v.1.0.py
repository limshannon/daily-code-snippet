#!/usr/bin/env python
# coding: utf-8

# # PART 01) 파이썬 기초

# ## 1장. 파이썬 기초

# ### 1절. 파이썬 다루기

# 코드 없음

# 

# ### 2절. 파이썬 기초 중의 기초

# #### 1. 변수(variables)

# In[1]:


name = 'Jane' # name 변수에 Jane 이라는 문자 할당
number = 1 # number 변수에 숫자 1 할당


# #### 2. 변수명 규칙

# 코드 없음

# #### 3. 메소드(methods)

# 코드 없음

# #### 4. 기초 연산자

# ##### 가. 산술 연산자(Arithmetic operator)

# In[2]:


7 + 3 # 더하기


# In[4]:


7 - 3 # 빼기


# In[5]:


7 * 3 # 곱하기


# In[6]:


7 / 3 # 나누기


# In[7]:


7 % 3 # 나머지


# In[8]:


7 // 3 # 몫


# In[9]:


7 ** 3 # 제곱


# ##### 나. 할당 연산자(Assignment operator)

# In[10]:


a = 7 # a에 7 할당
a


# In[11]:


a = 7 # a에 7 할당
a += 3 # a에 할당된 7에 3을 더한 값을 재할당(a = a + 3과 같음)
a


# In[12]:


a = 7 # a에 7 할당
a -= 3 # a에 할당된 7에 3을 뺀 값을 재할당(a = a - 3과 같음)
a


# In[13]:


a = 7 # a에 7 할당
a *= 3 # a에 할당된 7에 3을 곱한 값을 재할당(a = a * 3과 같음)
a


# In[14]:


a = 7 # a에 7 할당
a /= 3 # a에 할당된 7에 3을 나눈 후 나눈 값을 재할당(a = a / 3과 같음)
a


# In[15]:


a = 7 # a에 7 할당
a //= 3 # a에 할당된 7에 3을 나눈 후 몫을 재할당(a = a // 3과 같음)
a


# In[16]:


a = 7 # a에 7 할당
a %= 3 # a에 할당된 7에 3을 나눈 후 나머지를 재할당(a = a % 3과 같음)
a


# In[17]:


a = 7 # a에 7 할당
a **= 3 # a에 할당된 7에 3 제곱한 값을 재할당(a = a ** 3과 같음)
a


# ##### 다. 비교 연산자(Comparison operator)

# In[18]:


a = 7; b = 3 # a와 b에 각각 7과 3을 할당


# In[19]:


a == b # a와 b가 같은 지에 대한 여부


# In[20]:


a != b # a와 b가 다른 지에 대한 여부


# In[21]:


a > b # a가 b보다 큰 지에 대한 여부


# In[22]:


a < b # a가 b보다 작은 지에 대한 여부


# In[23]:


a >= b # a가 b보다 크거나 같은 지에 대한 여부


# In[24]:


a <= b # a가 b보다 작거나 같은 지에 대한 여부


# ##### 라. 논리 연산자(Logical operatror)

# In[25]:


a = 7; b = 3; c = 5; d = 1 # 할당


# In[26]:


# a > b (True)이고 c < d (False)


# In[27]:


# 모두 참(True)이면 참(True)을 반환
(a > b) and (c < d)


# In[28]:


# 하나라도 참(True)이면 참(True)을 반환
(a > b) or (c < d)


# In[29]:


# 참(True)이면 거짓(False)을 반환
not(a > b)


# In[30]:


# 거짓(False)이면 참(True)을 반환
not(c < d)


# ##### 마. 기타연산자

# 코드 없음

# ##### 바. 연산자 우선순위

# 코드 없음

# 

# ### 3절. 데이터 타입

# #### 1. 숫자형(Number)

# In[176]:


# a에 정수 1 할당
a = 1
print(a)

# b에 실수 2.7 할당
b = 2.7 
print(b)


# #### 2. 시퀀스형(Sequence)

# ##### 가. 문자열(String)

# In[31]:


# 방법1. 큰 따옴표("")로 둘러싸기
dataedu_1 = "BigData, A.I Technology Expert Group"
print(dataedu_1)

# 방법2. 작은 따옴표('')로 둘러싸기
dataedu_2 = 'BigData, A.I Technology Expert Group'
print(dataedu_2)

# 방법3. 큰 따옴표 연속 3개로 둘러싸기
dataedu_3 = """BigData, A.I Technology Expert Group"""
print(dataedu_3)

# 방법4. 작은 따옴표 연속 3개로 둘러싸기
dataedu_4 = '''BigData, A.I Technology Expert Group'''
print(dataedu_4)

# (참고) 작은 따옴표 연속 3개로 둘러싸는 방법은 줄바꿈까지 허용함
dataedu_5 = '''BigData,
A.I Technology Expert Group'''
print(dataedu_5)

# (참고) 큰 따옴표 연속 3개로 둘러싸는 방법은 줄바꿈까지 허용함
dataedu_6 = """BigData,
A.I Technology Expert Group"""
print(dataedu_6)

# (참고) ''를 포함하는 문자열은 ""로 둘러싸는 방법으로 입력이 가능함
dataedu_7 = "BigData, 'A.I Technology Expert Group'"
print(dataedu_7) # 결과 출력

# (참고) ""를 포함하는 문자열은 ''로 둘러싸는 방법으로 입력이 가능함
dataedu_8 = 'BigData, "A.I Technology Expert Group"'
print(dataedu_8) # 결과 출력


# ##### 나. 리스트(List)

# In[32]:


# 리스트 생성(숫자일 경우)
list_1 = [1, 2, 3, 4]
print(list_1)

# 리스트 생성(문자일 경우)
list_2 = ['D','a','T','a','E','d','u']
print(list_2)

# 리스트 생성(숫자+문자일 경우) 
list_3 = [1, 2, 3, 'E','d','u']
print(list_3)

# 빈 리스트 생성 (대괄호 이용)
list_4 = []

# 빈 리스트 생성 (list() 함수 이용)
list_5 = list() # list() 함수로 빈 리스트 생성


# ##### 다. 튜플(Tuple)

# In[33]:


# 튜플 생성(숫자일 경우)
tuple_1 = (1, 2, 3, 4)
print(tuple_1)

# 튜플 생성(문자일 경우)
tuple_2 = ('D','a','T','a','E','d','u')
print(tuple_2)

# 튜플 생성(숫자+문자일 경우) 
tuple_3 = (1, 2, 3, 'E','d','u')
print(tuple_3)

# 소괄호로 빈 튜플 생성
tuple_4 = ()

# tuple() 함수로 빈 튜플 생성
tuple_5 = tuple()


# In[34]:


# 리스트 생성(리스트일 경우)
list_4 = [list_1, list_2]
print(list_4)

# 리스트 생성(튜플일 경우)
list_5 = [tuple_1, tuple_2]
print(list_5)

# 리스트 생성(리스트 + 튜플일 경우)
list_6 = [list_1, tuple_2]
print(list_6)

# 튜플 생성(튜플일 경우)
tuple_4 = (tuple_1, tuple_2)
print(tuple_4)

# 튜플 생성(리스트일 경우)
tuple_5 = (list_1, list_2)
print(tuple_5)

# 튜플 생성(리스트 + 튜플일 경우)
tuple_6 = (list_1, tuple_2)
print(tuple_6)


# ##### 라. range

# In[35]:


# range(start, end, step)
# 0에서 9까지 1 간격으로 증가(즉 0 ~ 9 생성)
range_1 = range(0, 10) 
range_2 = range(10) 

# step을 음수로 지정할 경우 감소
# 10에서 1까지 1 간격으로 감소(즉 10 ~ 1 생성)
range_3 = range(10, 0, -1)

# 빈 range 형태
range_4 = range(0)


# In[36]:


# 예시1 : 0에서 시작해 2 간격으로 증가하고 10을 포함하지 않는 숫자열
# 즉 0, 2, 4, 6, 8 생성
range_5 = range(0, 10, 2)
print(list(range_5)) # 리스트로 변환하여 출력

# 예시2 : 2에서 시작해 3 간격으로 증가하고 17을 포함하지 않는 숫자열
# 즉 2, 5, 8, 11, 14 생성
range_6 = range(2, 17, 3) 
print(list(range_6)) # 리스트로 변환하여 출력

# 예시3 : 15에서 시작해 2 간격으로 감소하고 3을 포함하지 않는 숫자열
# 즉 15, 13, 11, 9, 7, 5 생성
range_7 = range(15, 3, -2)
print(list(range_7)) # 리스트로 변환하여 출력


# ##### 마. 시퀀스형 연산

# ###### 1) 인덱싱

# In[37]:


# 시퀀스형(문자열, 리스트, 튜플, range) 연산
String = 'DataEdu, Python!'
List = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
Range = range(0, 10)

# 문자열 인덱싱
print(String[0]) # 0번부터 시작, 제일 앞에서 한 글자
print(String[-1]) # 뒤에서부터 시작, 제일 뒤에서 한 글자
print(String[10]) # 앞에서부터 10번에서 한 글자 
print(String[-6]) # 뒤에서부터 6번에서 한 글자 


# In[38]:


# 리스트 인덱싱
print(List[0]) # 0번부터 시작, 제일 앞에서 한 원소
print(List[-1]) # 뒤에서부터 시작, 제일 뒤에서 한 원소
print(List[7]) # 앞에서부터 7번에서 한 원소
print(List[-2]) # 뒤에서부터 2번에서 한 원소


# In[39]:


# 튜플 인덱싱
print(Tuple[0]) # 0번부터 시작, 제일 앞에서 한 원소
print(Tuple[-1]) # 뒤에서부터 시작, 제일 뒤에서 한 원소
print(Tuple[7]) # 앞에서부터 7번에서 한 원소
print(Tuple[-2]) # 뒤에서부터 2번에서 한 원소


# In[40]:


# range 인덱싱
print(Range[0]) # 0번부터 시작, 제일 앞에서 한 숫자
print(Range[-1]) # 뒤에서부터 시작, 제일 뒤에서 한 숫자
print(Range[7]) # 앞에서부터 7번에서 한 숫자
print(Range[-2]) # 뒤에서부터 2번에서 한 숫자


# In[41]:


# 인덱싱을 통한 리스트의 특정 객체 수정
List = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# "변수명[k] = 원하는값"
List[0] = -1 # 0을 –1로 교체
List[-1] = 100 # 9를 100으로 교체
print(List)


# In[42]:


# 인덱싱을 통한 문자열의 특정 객체 수정(불가능)
String = 'DataEdu, Python!'
String[0] = 'd' # 'D'를 'd'로 교체


# In[43]:


# 인덱싱을 통한 튜플의 특정 객체 수정(불가능)
Tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
Tuple[0] = -1 # 0을 –1로 교체


# In[190]:


# 인덱싱을 통한 range의 특정 객체 수정(불가능)
Range = range(0, 10)
Range[0] = -1 # 0을 –1로 교체


# ###### 2) 슬라이싱

# In[191]:


# 시퀀스형(문자열, 리스트, 튜플, range) 슬라이싱
# 문자열 슬라이싱
String = 'DataEdu, Python!'

# 0번과 7번 사이의 객체를 추출함
print(String[0:7])
print(String[:7])

# 9번과 16번 사이의 객체를 추출함
print(String[9:])
print(String[9:16])

# 4번과 7번 사이의 객체를 추출함
print(String[4:7])


# In[44]:


# 리스트 슬라이싱
List = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 0번과 2번 사이의 객체를 추출함
print(List[0:2])
print(List[:2])

# 6번과 10번 사이의 객체를 추출함
print(List[6:])
print(List[6:10])

# 3번과 7번 사이의 객체를 추출함
print(List[3:7])


# In[45]:


# 튜플 슬라이싱
Tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

# 0번과 2번 사이의 객체를 추출함
print(Tuple[0:2])
print(Tuple[:2])

# 6번과 10번 사이의 객체를 추출함
print(Tuple[6:])
print(Tuple[6:10])

# 3번과 7번 사이의 객체를 추출함
print(Tuple[3:7])


# In[46]:


# range 슬라이싱
Range = range(0, 10)
# 0번과 2번 사이의 객체를 추출함
print(Range[0:2])
print(Range[:2])

# 6번과 10번 사이의 객체를 추출함
print(Range[6:])
print(Range[6:10])

# 3번과 7번 사이의 객체를 추출함
print(Range[3:7])


# ###### 3) 연결

# In[47]:


# 문자열 연결
'Data' + 'Edu'


# In[48]:


# 리스트 연결
[1, 2, 3] + [4, 5, 6]


# In[49]:


# 튜플 연결
(1, 2, 3) + (4, 5, 6)


# ###### 4)  반복

# In[50]:


# 문자열 반복
'Data' * 3


# In[51]:


# 리스트 반복
[1, 2, 3] * 3


# In[52]:


# 튜플 반복
(1, 2, 3) * 3


# ###### 5) 길이 정보

# In[53]:


# 문자열 길이정보
String = 'DataEdu, Python!'
len(String)


# In[54]:


# 리스트 길이정보
List = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
len(List)


# In[55]:


# 튜플 길이정보
Tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
len(Tuple)


# In[56]:


# range 길이정보
Range = range(0, 10)
len(Range)


# ##### 바. 시퀀스형 주요 내장함수와 메소드

# In[57]:


# 내장함수
# 시퀀스형 생성 후 할당
Str = 'dbac' # 문자열
List = [5, 1, 3, 9] # 리스트
Tuple = (6, 10, 3, 4) # 튜플
Range = range(2,13,3) # range 2, 5, 8, 11

# 내장함수 sum
# 시퀀스객체(문자열 제외)의 요소의 합을 반환
print(sum(List))
print(sum(Tuple))
print(sum(Range))

# 내장함수 max
# 시퀀스객체의 요소 중 가장 큰 요소 반환
print(max(Str))
print(max(List))
print(max(Tuple))
print(max(Range))

# 내장함수 min
# 시퀀스객체의 요소 중 가장 작은 요소 반환
print(min(Str))
print(min(List))
print(min(Tuple))
print(min(Range))

# 내장함수 sorted
# 시퀀스객체 정렬 후 리스트로 반환
print(sorted(Str)) # 리스트로 반환됨
print(sorted(List)) # 리스트로 반환됨
print(sorted(Tuple)) # 리스트로 반환됨
print(sorted(Range)) # 리스트로 반환됨


# In[59]:


# 내장함수 reversed
# 시퀀스객체의 요소 순서를 역순으로 반환
# reversed는 단순히 순서만을 역순으로 반환하는 것임
print(reversed(Str))
print(reversed(List))
print(reversed(Tuple))
print(reversed(Range))

# reversed할 경우 reverse 객체가 되므로 안의 요소를 확인할 수 없음
# 리스트로 변환하면 결과를 확인할 수 있음
print(list(reversed(Str)))
print(list(reversed(List)))
print(list(reversed(Tuple)))
print(list(reversed(Range)))


# In[60]:


# del 함수는 리스트객체의 특정 요소를 삭제함
del List[1]
print(List)


# In[61]:


# 문자열과 튜플, range에서 del을 사용할 수 없음을 확인
del Str[1]


# In[62]:


del Tuple[1]


# In[63]:


del Range[1]


# In[64]:


# 메소드
# 시퀀스형 생성 후 할당
Str = 'dbbac' # 문자열
List = [5, 1, 3, 3, 9] # 리스트
Tuple = (6, 10, 3, 4, 10) # 튜플
Range = range(2,13,3) # range 2, 5, 8, 11


# In[65]:


# 메소드 .index(요소)
# 시퀀스객체 내 해당 요소의 위치 인덱스 번호
Str.index('a')


# In[66]:


List.index(1)


# In[67]:


Tuple.index(3)


# In[68]:


Range.index(11)


# In[69]:


# 메소드 .count(요소)
# 시퀀스객체 내 해당 요소의 빈도


# In[70]:


Str.count('b')


# In[71]:


List.count(3)


# In[72]:


Tuple.count(10)


# In[73]:


Range.count(11)


# In[74]:


# 리스트 생성 후 할당
List = [5, -1, 6, 7, 1, 3]

# 메소드 .append(요소)
# 리스트객체의 맨 뒤에 요소를 추가
List.append(2) # 맨 뒤에 요소 2 추가
print(List)

# .insert(인덱스, 요소)
# 리스트객체의 특정 인덱스에 요소 추가
List.insert(3, 8)  # 3번 인덱스에 8 추가
print(List)

# .remove(요소)
# 리스트객체에서 처음으로 나오는 요소를 제거
List.remove(8) # 처음으로 나오는 8을 제거
print(List)

# .pop()
# 리스트객체의 마지막 요소 삭제 후 마지막 요소 반환
List_2 = List.pop()
print(List)
print(List_2)

# .sort(reverse=False)
# 리스트객체의 요소를 오름차순으로 정렬
List.sort()
print(List)

# reverse=True는 내림차순
List.sort(reverse=True)
print(List)

# .reverse()
# 리스트객체의 요소 순서를 역순으로 대치
List = [5, -1, 6, 7, 1, 3] # 리스트 생성 후 할당
List.reverse()
print(List)


# In[75]:


# 리스트 생성 후 할당
List = [5, 9, 6, 7, 1, 3]

# 메소드 .copy()
List_1 = List.copy()
print(List_1)

List_1[1] = -1 # 인덱싱을 통한 요소 교체
print(List_1) # 1번 인데스가 –1로 교체됨
print(List) # 원래 리스트에 변화 없음

# .copy() 안 할 경우
# 할당연산자 사용
List_2 = List
print(List_2)

List_2[1] = -1 # 인덱싱을 통한 요소 교체
print(List_2) # 1번 인덱스가 –1로 교체됨
print(List) # 원래 리스트도 변화함


# #### 3. 딕셔너리(Dictionary)

# In[76]:


# 딕셔너리
dic_1 = {'A':1, 'B':2, 'C':3} # {key : value}로 딕셔너리 생성
print(dic_1)

dic_2 = {1:'A', 2:'B', 3:'C'} # {key : value}로 딕셔너리 생성
print(dic_2)

dic_3 = dict(A = 1, B = 2, C = 3) # dict() 함수를 통한 방법 (key = value)
print(dic_3)


# In[77]:


# 딕셔너리에 키와 값 추가
dic_1['D'] = 4 # key D와 value 4를 추가
print(dic_1)

dic_2[4] = 'D' # key 4와 value D를 추가
print(dic_2)


# In[78]:


# 동일한 키와 다른 값을 추가할 경우 값만 교체됨
dic_1['A'] = 7
print(dic_1)


# In[79]:


# 딕셔너리에서의 조회
dic = {'A':1, 'B':2, 'C':3} # 딕셔너리 생성

# key를 통한 값 조회
print(dic['A'])
print(dic['C'])


# In[80]:


# 딕셔너리 타입의 주요내장함수
# del
# 딕셔너리객체에서 특정 키와 쌍인 값을 삭제
del dic['A']
print(dic)


# In[81]:


# .keys()
# 딕셔너리객체의 키로 구성된 리스트를 반환
dic.keys()


# In[82]:


# .values()
# 딕셔너리객체의 키로 구성된 리스트를 반환
dic.values()


# In[83]:


# .items()
# 딕셔너리객체의 키, 값 쌍으로 구성된 리스트를 반환
dic.items()


# In[84]:


# .update({키:값})
# 딕셔너리객체에 키, 값 쌍으로 수정하거나 추가함
dic.update({'D':7})
print(dic)


# #### 4. 집합(Set)

# In[85]:


# 집합
set_1 = {1, 2, 3, 4} # 중괄호로 집합 생성
set_2 = {'E', 'd', 'u'} # 중괄호로 집합 생성


# In[86]:


# 빈 집합은 중괄호만으로 생성 불가, 중괄호 사용 시 빈 딕셔너리를 생성함
# set()로만 빈 집합의 생성이 가능함
dic_1 = {} # 중괄호는 빈 딕셔너리 생성
set_1 = set() # set() 함수로 빈 집합(즉, 공집합) 생성


# In[87]:


# 집합 타입의 주요 메소드
set_3 = {2, 4, 6, 8, 10}
set_4 = {4, 8}


# In[88]:


# .add(원소)
# 집합객체에서 하나의 원소를 추가(순서를 보장하지 않으므로 결과가 다를 수 있음)
set_3.add(12)
print(set_3)


# In[89]:


# .update(원소)
# 집합객체에서 여러 개의 원소를 추가(순서를 보장하지 않으므로 결과가 다를 수 있음)
set_4.update({12, 16})
print(set_4)


# In[90]:


# .union(집합2)
# 집합객체1과 집합객체2의 합집합
set_3.union(set_4)


# In[91]:


#.intersection(집합2)
# 집합객체1과 집합객체2의 교집합
set_3.intersection(set_4)


# In[92]:


# .difference(집합2)
# 집합객체1과 집합객체2의 차집합
set_3.difference(set_4)


# #### 5. 부울형(Boolean)

# In[93]:


# 부울형 할당
a = True
b = False


# In[94]:


# 숫자형 -> 부울형 변환
print(bool(1), bool(0), bool())


# In[95]:


# 문자열 -> 부울형 변환
print(bool('A'), bool(''), bool())


# In[96]:


# 리스트 -> 부울형 변환
print(bool(['a', 'b']), bool([]))


# In[97]:


# 튜플 -> 부울형 변환
print(bool(('a', 'b')), bool(()))


# In[98]:


# 딕셔너리 -> 부울형 변환
print(bool({'a':1, 'b':2, 'c':3}), bool({}))


# In[99]:


# 집합 -> 부울형 변환
print( bool({'a', 'b'}), bool(set()) )


# #### 6. 데이터 타입 확인

# In[100]:


# 문자열
Str = 'abcd' 
print(type(Str))

# 리스트
List = [1, 2, 3, 4] 
print(type(List))

# 튜플
Tuple = (1, 2, 3, 4) 
print(type(Tuple))

# 딕셔너리
Dic = {'a':1, 'b':2, 'c':3} 
print(type(Dic))

# 집합
Set = {'a', 'b', 'c', 'd'} 
print(type(Set))

# 부울
Bool = True 
print(type(Bool))


# 

# ### 4절. 사용자 정의 함수

# #### 1. 함수 정의 문법

# 코드 없음

# #### 2. 함수 정의 예시

# In[101]:


# 하나의 숫자를 입력받아 제곱값을 계산하는 함수를 생성해보자.
# 함수 생성
def fun1(num) :
    return(num**2)

# fun1함수에 숫자 2를 입력
fun1(2)


# In[102]:


# 여러 숫자를 입력받아 합을 출력하는 함수를 생성해보자.
# 함수 생성
def fun2(*args) :
    x = sum(*args)
    print("합계 :", x)
    
# fun2 함수에 2,4,6,8,10을 입력
fun2([2,4,6,8,10])


# In[103]:


# 중첩함수 구문을 통해 두 개의 숫자를 입력받아 두 숫자와 그 합을 출력하는 함수를 생성해보자.
# 함수 생성
def fun3(x,y) :
    print(x)
    print(y)
    def fun4(*fun3) : # fun3 함수 내에 fun4 함수 정의 인자는 fun3의 인자 그대로
        Sum = x + y
        print(x, "+", y, "=", Sum)
    fun4(x,y)
    
fun3(3,5)


# 

# ### 5절. 제어문

# #### 1. 조건문

# In[267]:


# 정수를 입력받아 해당 숫자가 짝수인지 홀수인지 판단하고, 만약 입력받은 숫자가 정수가 아닐 경우 “정수를 입력해주세요.”라는 문구를 출력하는 if문을 작성해보자
x = 9
if x % 2 == 0 : # x를  2로 나눈 나머지가 0이면 참
    print(x, "는 짝수입니다.") # 조건(x % 2 == 0)이 참일 때 수행할 코드
elif x % 2 == 1 : # x를  2로 나눈 나머지가 1이면 참
    print(x, "는 홀수입니다.") # 조건(x % 2 == 1)이 참일 때 수행할 코드
else :
    print("정수를 입력해주세요.") # 모든 조건 거짓일 때 수행할 코드


# #### 2. 반복문

# ##### 1. for문

# In[268]:


# 반복문을 사용하지 않는 경우 : print문 4번 수행
print("The year is", 2015)
print("The year is", 2016)
print("The year is", 2017)
print("The year is", 2018)


# In[269]:


# 반복문을 사용하는 경우 year에 2015~2018이 모두 대입될 때까지 반복
for year in range(2015,2019) : #range(2015, 2019)는 2015 ~ 2018
    print("The year is", year)


# In[270]:


# year가 2018 이하인 동안 코드 수행
year = 2015 # 초기값

while year <= 2018 :
    print("The year is", year)
    year = year + 1  # 증가 코드


# -------------

# ## 2장. 패키지와 모듈

# ### 1절. 패키지와 모듈

# 코드 없음

# ### 2절. 패키지 소개

# 코드 없음

# ### 3절. 사용하기

# In[1]:


# 한 번만 실행

get_ipython().system('pip install numpy==1.21.1')
get_ipython().system('pip install pandas==1.4.2')
get_ipython().system('pip install scipy==1.7.0')
get_ipython().system('pip install scikit-learn==0.24.2')


# In[2]:


# 넘파이 불러오기
import numpy as np # 넘파이 패키지를 np라는 별명으로 불러옴

# 판다스 불러오기
import pandas as pd # 판다스 패키지를 pd라는 별명으로 불러옴

# 사이파이 불러오기
import scipy as sp # 사이파이 패키지를 sp라는 별명으로 불러옴

# 사이킷런 불러오기
import sklearn


# In[3]:


# 넘파이 정보 확인하기
get_ipython().system(' pip show numpy')


# In[4]:


# 판다스 정보 확인하기
get_ipython().system(' pip show pandas')


# In[5]:


# 사이파이 정보 확인하기
get_ipython().system(' pip show scipy')


# In[6]:


# 사이킷런 정보 확인하기
get_ipython().system(' pip show scikit-learn')


# # (끝)
