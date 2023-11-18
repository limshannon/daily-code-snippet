#0.5862051322865622
import pandas as pd
train = pd.read_csv("data/customer_train.csv")
test = pd.read_csv("data/customer_test.csv")
from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
le2 = LabelEncoder()
le.fit(train['주구매상품'])
le2.fit(train['주구매지점'])

print(train.info())
print(train.describe())
print(train.head(10))

train['주구매상품'] = le.transform(train['주구매상품'])
train['주구매지점'] = le2.transform(train['주구매지점'])
train['성별'] = train['성별'].astype('category')

train = train.drop('회원ID', axis=1)
test = test.drop('회원ID', axis=1)

test['주구매상품'] = le.transform(test['주구매상품'])
test['주구매지점'] = le2.transform(test['주구매지점'])


train['환불금액']= train['환불금액'].fillna(0)
test['환불금액'] = test['환불금액'].fillna(0)

import numpy as np
from sklearn.compose import make_column_selector, make_column_transformer
#mct = make_column_transformer(StandardScaler(), make_column_selector(dtype_include=np.number))
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
#res_df.to_csv('result.csv', index=False)
