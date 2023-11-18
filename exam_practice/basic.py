# 출력을 원할 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

import pandas as pd

a = pd.read_csv("data/mtcars.csv")

# 사용자 코딩

# 해당 화면에서는 제출하지 않으며, 문제 풀이 후 답안제출에서 결괏값 제출
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
target= a['qsec'].values.reshape(-1,1)
scaler.fit(target)
res = scaler.transform(target)
print(len(res[res>0.5]))
 
#print(narray.where(>=0.5))
#print(help(MinMaxScaler))

#2백화점 고객 데이터로 성별예측하기.
#data/customer_train.csv
#data/customer_test.csv
