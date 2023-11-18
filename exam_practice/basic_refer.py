# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

# 데이터 파일 읽기 예제
import pandas as pd
train = pd.read_csv("data/customer_train.csv")
test = pd.read_csv("data/customer_test.csv")
Y = train['성별']
X = train.drop('성별', axis=1)

X_submission = pd.read_csv("data/customer_test.csv")

dfX = pd.concat([X, X_submission])
dfX.info()
dfX = pd.concat([X, X_submission], ignore_index = True)
temp = X.groupby('주구매상품')['환불금액'].transform('mean')
dfX['환불금액'] = dfX['환불금액'].mask(dfX['환불금액'].isna(),temp) 

# dfX['환불금액']의 결측치를 temp로 채우기
dfX['환불금액'] = dfX['환불금액'].fillna(dfX['환불금액'].mean())
dfX.isna().sum().sum()

dfX['주구매지점'] = dfX['주구매지점'].astype('category').cat.codes
dfX['주구매상품'] = dfX['주구매상품'].astype('category').cat.codes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

def get_scores(model, xtrain, xtest, ytrain, ytest):
	A = model.score(xtrain,ytrain)
	B = model.score(xtest,ytest)
	ypred = model.predict(xtest)#[:,1]
	C = roc_auc_score(ytest,ypred)

	return '{:.4f}  {:.4f}  {:.4f}'.format(A,B,C)

def make_models(xtrain, xtest, ytrain, ytest):
	model1 = LogisticRegression(max_iter=500).fit(xtrain, ytrain)
	print('model1', get_scores(model1, xtrain, xtest, ytrain, ytest))
	
	for k in range(1,10):
		model2 = KNeighborsClassifier(k).fit(xtrain,ytrain)
		print('model2', k, get_scores(model2, xtrain, xtest, ytrain, ytest))

	model3 = DecisionTreeClassifier(random_state=0).fit(xtrain,ytrain)
	print('model3', get_scores(model3, xtrain, xtest, ytrain, ytest))
	for d in range(3,8):
		model3 = DecisionTreeClassifier(max_depth=d, random_state=0).fit(xtrain,ytrain)
		print('model3', d, get_scores(model3, xtrain, xtest, ytrain, ytest))

	model4 = RandomForestClassifier(random_state=0).fit(xtrain,ytrain)
	print('model4', get_scores(model4, xtrain, xtest, ytrain, ytest))
	
	for rf in range(3,8):
		model3 = DecisionTreeClassifier(max_depth=rf, random_state=0).fit(xtrain,ytrain)
		print('model4', rf, get_scores(model3, xtrain, xtest, ytrain, ytest))

	model5 = XGBClassifier(eval_metric='logloss', use_label_encoder=False).fit(xtrain,ytrain)
	print('model4', get_scores(model5, xtrain, xtest, ytrain, ytest))

def get_data(dfX,Y):
	X = dfX.drop(columns=['회원ID'])
	X_use = X.iloc[:3500,:]
	X_submission = X.iloc[3500:,:]
	Y1 = Y 
	scaler = StandardScaler()
	X1_use = scaler.fit_transform(X_use)
	X1_submission = scaler.transform(X_submission)
	print(X1_use.shape, X1_submission.shape, Y1.shape)
	return X1_use, X1_submission, Y1

# 분리하기
X1_use, X1_submission, Y1 = get_data(dfX,Y)
xtrain, xtest, ytrain, ytest = train_test_split(X1_use, Y1, test_size=0.3, stratify=Y1, random_state=0)
make_models(xtrain, xtest, ytrain, ytest)

model = RandomForestClassifier(500, max_depth=6,random_state=0).fit(xtrain, ytrain)
print('final model', get_scores(model, xtrain, xtest, ytrain, ytest))

print(X_submission.columns)
pred = model.predict(X_submission.drop('회원ID',axis=1) )#[:,1])
print(pred)
submission = pd.DataFrame({'cust_id':X_submission['회원ID'],
							'성별':pred})
submission.to_csv('submission.csv', index = False)
