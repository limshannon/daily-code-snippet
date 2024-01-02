'''
'''
import pandas as pd
import numpy as np
df = pd.read_csv("data/Titanic.csv")
#['PassengerId', 'Survived', 'Pclass', 'Name', 'Gender', 'Age', 'SibSp',        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],      dtype='object')

from scipy.stats import chisquare, chi2_contingency
#print(help(chisquare))
#print(help(chi2_contingency))

'''
1) Gender와 Survived 변수 간의 독립성 검정을 실시하였을 때, 카이제곱 통계량은? (반올림하여 소수 셋째 자리까지 계산)
'''
print(chisquare(df['Gender'].value_counts()))
print(chisquare(df['Survived'].value_counts()))
freq_table = pd.crosstab(df['Gender'], df['Survived'])
"""
즉, 대부분의 여성들은 사고에서 살아남았다.
그리고 대부분의 남자들이 죽었습니다.
그러나 이 차이는 얼마나 중요한가?
"""
chi2stat, p, dof, freq_exp = chi2_contingency(freq_table)
print(chi2stat)
print(round(chi2stat,3))

'''
2) Gender, SibSp, Parch, Fare를 독립변수로 사용하여 로지스틱 회귀모형을 실시하였을 때, Parch변수의 계수값은?
'''
from statsmodels.formula.api import logit
formular_str = "Survived ~ C(Gender) + SibSp + Parch + Fare"
model = logit(formular_str, df).fit()
print(model.params)
print(round(model.params['Parch'],3))

'''
3) 위 2번 문제에서 추정된 로지스틱 회귀모형에서 SibSp변수가 한 단위 증가할 때 생존할 오즈비(Odds Ratio)값은?
'''

odds = np.exp(model.params)