#!/usr/bin/env python
# coding: utf-8

# ### PART 04) 모델링

# ## 1장. 지도학습

# ### 1절. 데이터분할

# #### 1. 홀드아웃

# In[5]:


# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target


# In[6]:


from sklearn.model_selection import train_test_split

# arrays에 아래와 같이 data와 target을 둘 다 넣을 경우,
# X와 y에 대해 train과 test가 분할된 데이터셋들을 반환함
# cf) data만 입력하면 X에 대한 train, test를 분할해서 반환함
# random_state를 특정 숫자로 입력할 경우, 계속해서 동일한 데이터셋으로 분할됨   
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.3, #7:3
                                                    random_state = 2022)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[7]:


# stratify = target은 원래의 target 컬럼의 0과 1의 비율을 반영하여 데이터를 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.3, #7:3
                                                    random_state = 2022,
                                                    stratify = target)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# #### 2. K-fold

# In[8]:


# 넘파이 배열 생성
import numpy as np
X = np.arange(10)

# KFold 클래스 호출
from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5) # k = 5

# 메소드 .split은 학습, 평가 데이터의 인덱스를 생성해줌
for train_idx, test_idx in kfold.split(X) :
    print("학습 :", train_idx, "평가 :", test_idx) # 인덱스번호


# In[9]:


# 넘파이 배열 생성
import numpy as np
X = np.arange(15) 
y = [0] * 6 + [1] * 3 + [2] * 6 #리스트 생성
# y = [0,0,0,0,0,0,1,1,1,2,2,2,2,2,2]로 해도 됨

# StratifiedKFold 클래스 호출
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits = 3) # k = 3

# 메소드 .split은 학습, 평가 데이터의 인덱스를 생성해줌
# 동시에 y의 0,1,2 비율도 함께 고려함
for train_idx, test_idx in kfold.split(X, y) :
     print("학습 :", train_idx, "평가 :", test_idx) # 인덱스번호


# ### 2절. 성과분석

# #### 1. 분류 지표

# ##### 가. 혼동 행렬을 이용한 평가 지표

# In[10]:


# 함수 confusion_matrix() 호출
from sklearn.metrics import confusion_matrix

# 이진분류
y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 1, 0, 1, 1, 1]

confusion_matrix(y_true, y_pred) # 혼동행렬


# In[11]:


# 이진분류(레이블로 되어있을 경우)
y_true = ['A', 'A', 'A', 'B', 'B', 'B']
y_pred = ['A', 'B', 'A', 'B', 'B', 'B']

confusion_matrix(y_true, y_pred, labels = ['A', 'B']) # 혼동행렬(레이블구분)


# In[12]:


# 다지분류(레이블:0,1,2)
y_true = [0, 0, 0, 1, 1, 2, 2, 2, 2]
y_pred = [0, 1, 1, 1, 0, 0, 1, 2, 2]

confusion_matrix(y_true, y_pred) # 혼동행렬


# In[13]:


# 함수 호출
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 이진분류
y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 1, 0, 1, 1, 1]

# 정분류율(Accuracy)
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)

# 재현율(Recall)
recall = recall_score(y_true, y_pred)
print(recall)

# 정밀도(Precision) 
precision = precision_score(y_true, y_pred)
print(precision)

# f1-score
f1 = f1_score(y_true, y_pred)
print(f1)


# In[14]:


# 함수 호출
from sklearn.metrics import roc_curve, auc

# 이진분류
y_true = [0, 0, 0, 1, 1, 1]
y_score = [0.1, 0.75, 0.35, 0.92, 0.81, 0.68]

# ROC
# 함수 roc_curve()는 fpr, tpr, thresholds 세 가지를 반환함
fpr, tpr, thresholds = roc_curve(y_true, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# #### 2. 예측 지표

# In[15]:


# 함수 호출
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# 연속형 데이터
# 균일분포 (0,1)에서 임의의 난수 생성
import numpy as np
np.random.seed(123) # 난수 고정

y_true = np.random.random_sample(5) # 균일분포 (0,1)에서 5개 랜덤 추출
print(y_true)

y_pred = np.random.random_sample(5) # 균일분포 (0,1)에서 5개 랜덤 추출
print(y_pred)

# MSE
mse = mean_squared_error(y_true, y_pred)
print(mse)

# MAE
mae = mean_absolute_error(y_true, y_pred)
print(mae)

# MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)
print(mape)


# ### 3절. 로지스틱 회귀분석

# In[16]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
lr_bin = LogisticRegression(C = 0.5, # 규제 강도
                            max_iter = 2000) # 수렴까지 걸리는 최대 반복 횟수
 
# 모델학습
model_lr_bin = lr_bin.fit(X_train, y_train)

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_lr_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[17]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

 
# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
lr_multi = LogisticRegression(C = 0.05, # 규제 강도(default = 1.5)
                              max_iter = 200) # 수렴까지 걸리는 최대 반복 횟수

# 모델학습
model_lr_multi = lr_multi.fit(X_train, y_train)

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_lr_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# ### 4절. 서포트벡터머신

# In[18]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
# 메소드 .predict_proba()의 사용을 위해서 probability = True 입력 필요
svm_bin = SVC(kernel = 'linear', C = 0.5, probability = True) 

# 모델학습
model_svm_bin = svm_bin.fit(X_train, y_train)

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_svm_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[20]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
svm_multi = LinearSVC(C = 0.1)

# 모델학습
model_svm_multi = svm_multi.fit(X_train, y_train)

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_svm_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# In[21]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import train_test_split

# diabetes 데이터셋 호출
from sklearn.datasets import load_diabetes
diabetes = load_diabetes() 
data = diabetes.data
target = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205)

# 모형객체 생성
svm_conti_1 = SVR(C = 0.1, epsilon = 0.01)
svm_conti_2 = LinearSVR(C = 0.1, loss = 'squared_epsilon_insensitive')

# 모델학습
model_svm_conti_1 = svm_conti_1.fit(X_train, y_train)
model_svm_conti_2 = svm_conti_2.fit(X_train, y_train)

# RMSE
from sklearn.metrics import mean_squared_error

y_pred_1 = model_svm_conti_1.predict(X_test)
rmse_1 = mean_squared_error(y_test, y_pred_1, squared = False)
print(rmse_1)

y_pred_2 = model_svm_conti_2.predict(X_test)
rmse_2 = mean_squared_error(y_test, y_pred_2, squared = False)
print(rmse_2)


# ### 5절. 나이브베이즈

# In[22]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

 
# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
nb_bin = BernoulliNB(alpha = 0.5)

# 모델학습
model_nb_bin = nb_bin.fit(X_train, y_train)

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_nb_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[23]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
nb_multi = MultinomialNB(alpha = 1.5)

# 모델학습
model_nb_multi = nb_multi.fit(X_train, y_train)

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_nb_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# In[24]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


# diabetes 데이터셋 호출
from sklearn.datasets import load_diabetes
diabetes = load_diabetes() 
data = diabetes.data
target = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205)

# 모형객체 생성
nb_conti = GaussianNB()

# 모델학습
model_nb_conti = nb_conti.fit(X_train, y_train)

# RMSE
from sklearn.metrics import mean_squared_error
y_pred = model_nb_conti.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# ### 6절. K-최근접이웃

# In[25]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
 
# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
knn_bin = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')

# 모델학습
model_knn_bin = knn_bin.fit(X_train, y_train)

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_knn_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[26]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
knn_multi = KNeighborsClassifier(n_neighbors = 10, weights = 'distance')

# 모델학습
model_knn_multi = knn_multi.fit(X_train, y_train)

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_knn_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# In[27]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# diabetes 데이터셋 호출
from sklearn.datasets import load_diabetes
diabetes = load_diabetes() 
data = diabetes.data
target = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205)

# 모형객체 생성
knn_conti = KNeighborsRegressor(n_neighbors = 5, weights = 'distance')

# 모델학습
model_knn_conti = knn_conti.fit(X_train, y_train)

# RMSE
from sklearn.metrics import mean_squared_error
y_pred = model_knn_conti.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# ### 7절. 인공신경망

# In[28]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
ann_bin = MLPClassifier(alpha = 0.5,
                        max_iter = 500,
                        random_state = 2022)

# 모델학습
model_ann_bin = ann_bin.fit(X_train, y_train)

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_ann_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[29]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모델객체 생성
ann_multi = MLPClassifier(random_state = 2022, max_iter = 600)

# 모델학습
model_ann_multi = ann_multi.fit(X_train, y_train)

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_ann_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# In[30]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# diabetes 데이터셋 호출
from sklearn.datasets import load_diabetes
diabetes = load_diabetes() 
data = diabetes.data
target = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205)

# 모형객체 생성
ann_conti = MLPRegressor(alpha = 0.5,
                         max_iter = 10000,
                         random_state = 2022)

# 모델학습
model_ann_conti = ann_conti.fit(X_train, y_train)

# RMSE
from sklearn.metrics import mean_squared_error
y_pred = model_ann_conti.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# ### 8절. 의사결정나무

# In[31]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
dtr_multi = DecisionTreeClassifier(max_depth = 3,
                                   min_samples_leaf = 10,
                                   random_state = 2022)

# 모델학습
model_dtr_multi = dtr_multi.fit(X_train, y_train)

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_dtr_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# In[32]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
dtr_bin = DecisionTreeClassifier(max_depth = 3,
                                min_samples_leaf = 10,
                                random_state = 2022)

# 모델학습
model_dtr_bin = dtr_bin.fit(X_train, y_train)

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_dtr_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[34]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# diabetes 데이터셋 호출
from sklearn.datasets import load_diabetes
diabetes = load_diabetes() 
data = diabetes.data
target = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205)

# 모형객체 생성
dtr_conti = DecisionTreeRegressor(max_depth = 3,
                                  min_samples_leaf = 10,
                                  random_state = 2022)

# 모델학습
model_dtr_conti = dtr_conti.fit(X_train, y_train)

# RMSE
from sklearn.metrics import mean_squared_error
y_pred = model_dtr_conti.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# ### 9절. 앙상블

# #### 1. 배깅

# In[35]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
 
# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
dtr = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 10)

bag_bin = BaggingClassifier(base_estimator = dtr,
                            n_estimators = 500,
                            random_state = 2022)

# 모델학습
model_bag_bin = bag_bin.fit(X_train, y_train)

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_bag_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[36]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
dtr = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 10)

bag_multi = BaggingClassifier(base_estimator = dtr,
                              n_estimators = 500,
                              random_state = 2022)

# 모델학습
model_bag_multi = bag_multi.fit(X_train, y_train)

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_bag_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# In[37]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# diabetes 데이터셋 호출
from sklearn.datasets import load_diabetes
diabetes = load_diabetes() 
data = diabetes.data
target = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205)

# 모형객체 생성
dtr = DecisionTreeRegressor(max_depth = 3, min_samples_leaf = 10)

bag_conti = BaggingRegressor(base_estimator = dtr,
                              n_estimators = 500,
                              random_state = 2022)

# 모델학습
model_bag_conti = bag_conti.fit(X_train, y_train)

# RMSE
from sklearn.metrics import mean_squared_error
y_pred = model_bag_conti.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# #### 2. 랜덤포레스트

# In[38]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

 
# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
rf_bin = RandomForestClassifier(n_estimators = 500,
                                max_depth = 3,
                                min_samples_leaf = 10,
                                max_features = 'sqrt',
                                random_state = 2022)

# 모델학습
model_rf_bin = rf_bin.fit(X_train, y_train)

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_rf_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[39]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
rf_multi = RandomForestClassifier(n_estimators = 500,
                                  max_depth = 3,
                                  min_samples_leaf = 15,
                                  max_features = 'sqrt',
                                  random_state = 2022)

# 모델학습
model_rf_multi = rf_multi.fit(X_train, y_train)

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_rf_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# In[40]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# diabetes 데이터셋 호출
from sklearn.datasets import load_diabetes
diabetes = load_diabetes() 
data = diabetes.data
target = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205)

# 모형객체 생성
rf_conti = RandomForestRegressor(n_estimators = 500,
                                 max_depth = 3,
                                 min_samples_leaf = 10,
                                 max_features = 3,
                                 random_state = 2022)

# 모델학습
model_rf_conti = rf_conti.fit(X_train, y_train)

# RMSE
from sklearn.metrics import mean_squared_error
y_pred = model_rf_conti.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# #### 3. 에이다부스팅

# In[41]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

 
# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
ada_bin = AdaBoostClassifier(n_estimators = 100,
                             learning_rate = 0.5,
                             random_state = 2022)

# 모델학습
model_ada_bin = ada_bin.fit(X_train, y_train)

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_ada_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[42]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split


# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
ada_multi = AdaBoostClassifier(n_estimators = 500,
                               learning_rate = 0.01,
                               random_state = 2022)

# 모델학습
model_ada_multi = ada_multi.fit(X_train, y_train)

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_ada_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# In[43]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split


# diabetes 데이터셋 호출
from sklearn.datasets import load_diabetes
diabetes = load_diabetes() 
data = diabetes.data
target = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205)
# 모형객체 생성
ada_conti = AdaBoostRegressor(n_estimators = 500,
                              learning_rate = 0.01,
                              loss = 'square',
                              random_state = 2022)

# 모델학습
model_ada_conti = ada_conti.fit(X_train, y_train)

# RMSE
from sklearn.metrics import mean_squared_error
y_pred = model_ada_conti.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# #### 4. 그래디언트 부스팅

# In[44]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

 
# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
gbm_bin = GradientBoostingClassifier(n_estimators = 500,
                                     max_depth = 4,
                                     min_samples_leaf = 10,
                                     learning_rate = 0.1,
                                     random_state = 2022)

# 모델학습
model_gbm_bin = gbm_bin.fit(X_train, y_train)

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_gbm_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[45]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
gbm_multi = GradientBoostingClassifier(n_estimators = 500,
                                       max_depth = 8,
                                       min_samples_leaf = 5,
                                       learning_rate = 0.5,
                                       random_state = 2022)

# 모델학습
model_gbm_multi = gbm_multi.fit(X_train, y_train)

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_gbm_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# In[46]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# diabetes 데이터셋 호출
from sklearn.datasets import load_diabetes
diabetes = load_diabetes() 
data = diabetes.data
target = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205)

# 모형객체 생성
gbm_conti = GradientBoostingRegressor(n_estimators = 500,
                                      max_depth = 2,
                                      min_samples_leaf = 5,
                                      learning_rate = 0.5,
                                      random_state = 2022)

# 모델학습
model_gbm_conti = gbm_conti.fit(X_train, y_train)

# RMSE
from sklearn.metrics import mean_squared_error
y_pred = model_gbm_conti.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# #### 5. XGBoost

# In[20]:


get_ipython().system('pip install xgboost==1.4.2')


# In[47]:


# 패키지로부터 클래스, 함수를 호출
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

 
# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
xgb_wrap_bin = XGBClassifier(max_depth = 8,
                             n_estimators = 500,
                             nthread = 5,
                             min_child_weight = 20,
                             gamma = 0.5,
                             objective = 'binary:logistic',
                             use_label_encoder = False,
                             random_state = 2022)

# 모델학습
model_xgb_wrap_bin = xgb_wrap_bin.fit(X_train, y_train,
                                      eval_metric = 'mlogloss')

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_xgb_wrap_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[48]:


# 패키지로부터 클래스, 함수를 호출
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
xgb_wrap_multi = XGBClassifier(max_depth = 8,
                               n_estimators = 500,
                               nthread = 5,
                               min_child_weight = 10,
                               gamma = 0.5,
                               objective = 'multi:softmax',
                               use_label_encoder = False,
                               random_state = 2022)

# 모델학습
model_xgb_wrap_multi = xgb_wrap_multi.fit(X_train, y_train,
                                          eval_metric = 'mlogloss')

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_xgb_wrap_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# In[49]:


# 패키지로부터 클래스, 함수를 호출
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


# diabetes 데이터셋 호출
from sklearn.datasets import load_diabetes
diabetes = load_diabetes() 
data = diabetes.data
target = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205)

# 모형객체 생성
xgb_wrap_conti = XGBRegressor(max_depth = 8,
                              n_estimators = 500,
                              nthread = 5,
                              min_child_weight = 10,
                              gamma = 0.5,
                              objective = 'reg:squarederror',
                              use_label_encoder = False,
                              random_state = 2022)

# 모델학습
model_xgb_wrap_conti = xgb_wrap_conti.fit(X_train, y_train,
                                          eval_metric = 'rmse')

# RMSE
from sklearn.metrics import mean_squared_error
y_pred = model_xgb_wrap_conti.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# #### 6. LightGBM

# In[24]:


get_ipython().system('pip install lightgbm==3.3.2')


# In[50]:


# 패키지로부터 클래스, 함수를 호출
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

 
# breast_cancer 데이터셋 호출
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
data = breast_cancer.data
target = breast_cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
lgb_wrap_bin = LGBMClassifier(max_depth = 8,
                              n_estimators = 500,
                              n_jobs = 30,
                              min_child_weight = 10,
                              learning_rate = 0.2,
                              objective = 'binary',
                              random_state = 2022)

# 모델학습
model_lgb_wrap_bin = lgb_wrap_bin.fit(X_train, y_train)

# ROC
from sklearn.metrics import roc_curve, auc
y_score = model_lgb_wrap_bin.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score) 

# AUC
AUC = auc(fpr, tpr) # roc_curve()에서 반환된 fpr을 x축, tpr을 y축
print(AUC)


# In[51]:


# 패키지로부터 클래스, 함수를 호출
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
target = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205,
                                                    stratify = target)

# 모형객체 생성
lgb_wrap_multi = LGBMClassifier(max_depth = 8,
                                n_estimators = 500,
                                n_jobs = 5,
                                min_child_weight = 10,
                                learning_rate = 0.5,
                                objective = 'multiclass',
                                random_state = 2022
                               )


# 모델학습
model_lgb_wrap_multi = lgb_wrap_multi.fit(X_train, y_train)

# macro f1-score
from sklearn.metrics import f1_score
y_pred = model_lgb_wrap_multi.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = "macro")
print(macro_f1)


# In[52]:


# 패키지로부터 클래스, 함수를 호출
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


# diabetes 데이터셋 호출
from sklearn.datasets import load_diabetes
diabetes = load_diabetes() 
data = diabetes.data
target = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 2205)

# 모형객체 생성
lgb_wrap_conti = LGBMRegressor(max_depth = 8,
                               n_estimators = 500,
                               n_jobs = 5,
                               min_child_weight = 10,
                               learning_rate = 0.5,
                               objective = 'regression',
                               random_state = 2022)

# 모델학습
model_lgb_wrap_conti = lgb_wrap_conti.fit(X_train, y_train)

# RMSE
from sklearn.metrics import mean_squared_error
y_pred = model_lgb_wrap_conti.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# ## 2장. 군집 모형

# ### 1절. 군집 평가

# In[28]:


# 임의의 리스트 생성
labels_true = [0, 0, 0, 1, 1, 1, 1, 2, 2]
labels_pred = [0, 0, 1, 1, 1, 1, 2, 2, 2]

# 함수 호출
from sklearn.metrics.cluster import rand_score, adjusted_rand_score

# RI(랜드지수)
ri = rand_score(labels_true, labels_pred)
print(ri)

# ARI(조정 랜드지수)
ari = adjusted_rand_score(labels_true, labels_pred)
print(ari)


# ### 2절. 계층적 군집분석

# In[29]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import rand_score, adjusted_rand_score

# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data
labels_true = iris.target # 정답 레이블

# 와드연결법
agg_ward = AgglomerativeClustering(n_clusters = 3)
labels_pred_ward = agg_ward.fit_predict(data)

# 평균연결법
agg_avg = AgglomerativeClustering(n_clusters = 3, linkage = 'average')
labels_pred_avg = agg_avg.fit_predict(data)

# 최장연결법
agg_comp = AgglomerativeClustering(n_clusters = 3, linkage = 'complete')
labels_pred_comp = agg_comp.fit_predict(data)

# 최단연결법
agg_sing = AgglomerativeClustering(n_clusters = 3, linkage = 'single')
labels_pred_sing = agg_sing.fit_predict(data)


# In[30]:


# RI 비교
print(rand_score(labels_true, labels_pred_ward)) # 와드연결법
print(rand_score(labels_true, labels_pred_avg)) # 평균연결법
print(rand_score(labels_true, labels_pred_comp)) # 최장연결법
print(rand_score(labels_true, labels_pred_sing)) # 최단연결법


# In[31]:


# ARI 비교
print(adjusted_rand_score(labels_true, labels_pred_ward)) # 와드연결법
print(adjusted_rand_score(labels_true, labels_pred_avg)) # 평균연결법
print(adjusted_rand_score(labels_true, labels_pred_comp)) # 최장연결법
print(adjusted_rand_score(labels_true, labels_pred_sing)) # 최단연결법


# ### 3절. k-means 군집분석

# In[32]:


# 패키지로부터 클래스, 함수를 호출
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# iris 데이터셋 호출
from sklearn.datasets import load_iris
iris = load_iris() 
data = iris.data

# k=2일 때 k-means 군집모형으로 군집 형성
kmeas_k2 = KMeans(n_clusters = 2, random_state = 2022)
labels_pred_k2 = kmeas_k2.fit_predict(data)

# k=3일 때 k-means 군집모형으로 군집 형성
kmeas_k3 = KMeans(n_clusters = 3, random_state = 2022)
labels_pred_k3 = kmeas_k3.fit_predict(data)

# k=4일 때 k-means 군집모형으로 군집 형성
kmeas_k4 = KMeans(n_clusters = 4, random_state = 2022)
labels_pred_k4 = kmeas_k4.fit_predict(data)


# In[33]:


# 개체별로 연결볍에 따른 실루엣 계수를 계산
import pandas as pd # 데이터프레임 생성 필요

# k=2일 때
sil_k2 = silhouette_samples(data, labels_pred_k2) # 개체별 실루엣 계수

# 개체별 예측 레이블과 실루엣 계수를 각각 컬럼으로 가지는 데이터프레임 생성
df_k2 = pd.DataFrame({'labels' : labels_pred_k2, 'silhouette' : sil_k2})

# 레이블별 실루엣 계수의 평균
print(df_k2.groupby('labels')['silhouette'].mean())

# 전체 실루엣 계수 평균
print(silhouette_score(data, labels_pred_k2))


# In[34]:


# k=3일 때
sil_k3 = silhouette_samples(data, labels_pred_k3) # 개체별 실루엣 계수

# 개체별 예측 레이블과 실루엣 계수를 각각 컬럼으로 가지는 데이터프레임 생성
df_k3 = pd.DataFrame({'labels' : labels_pred_k3, 'silhouette' : sil_k3})

# 레이블별 실루엣 계수의 평균
print(df_k3.groupby('labels')['silhouette'].mean())

# 전체 실루엣 계수 평균
print(silhouette_score(data, labels_pred_k3))


# In[35]:


# k=4일 때
sil_k4 = silhouette_samples(data, labels_pred_k4) # 개체별 실루엣 계수

# 개체별 예측 레이블과 실루엣 계수를 각각 컬럼으로 가지는 데이터프레임 생성
df_k4 = pd.DataFrame({'labels' : labels_pred_k4, 'silhouette' : sil_k4})

# 레이블별 실루엣 계수의 평균
print(df_k4.groupby('labels')['silhouette'].mean())

# 전체 실루엣 계수 평균
print(silhouette_score(data, labels_pred_k4))


# # (끝)
