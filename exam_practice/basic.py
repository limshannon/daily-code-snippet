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

# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

# 데이터 파일 읽기 예제
import pandas as pd
train = pd.read_csv("data/customer_train.csv")
test = pd.read_csv("data/customer_test.csv")

train['주구매상품'] = train['주구매상품'].astype('category')
train['주구매지점'] = train['주구매지점'].astype('category')
train['성별'] = train['성별'].astype('category')

print(train.columns)
print(test.columns)

train = train.drop('회원ID', axis=1)
test = test.drop('회원ID', axis=1)
test['주구매상품'] = test['주구매상품'].astype('category')
test['주구매지점'] = test['주구매지점'].astype('category')

print(train.isna().sum())
print(test.isna().sum())
train['환불금액']= train['환불금액'].fillna(0)
test['환불금액'] = test['환불금액'].fillna(0)
print(train.isna().sum())
print(test.isna().sum())

x = train.drop('성별',axis=1)
y = train['성별']
x_encoded = pd.get_dummies(x)

from sklearn.model_selection import train_test_split
x_tr,x_val, y_tr,y_val = train_test_split(x_encoded, y, test_size=0.25)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_tr, y_tr) 

pred = rfc.predict(x_val)


from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_val, pred))

x_test_encoded = pd.get_dummies(test)
common_features = list(set(x_encoded.columns).intersection(x_test_encoded.columns))

x_encoded = x_encoded[common_features]
x_test_encoded = x_test_encoded[common_features]

rfc2 = RandomForestClassifier()
rfc2.fit(x_encoded, y)
print(x_encoded.columns)
print(x_test_encoded.columns)

pred2 = rfc2.predict(x_test_encoded)
res_df = pd.DataFrame({'pred': pred2})
print(res_df)
res_df.to_csv('result.csv', index=False)


