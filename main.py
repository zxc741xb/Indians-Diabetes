import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('D:\Kaggle\Indians_Diabetes\diabetes.csv')
# print(data.head())

"""
# 資料分布狀況
print(data.describe())

# Boxplot查看離群值
plt.boxplot(data['Pregnancies'], showmeans=True)
plt.title('Pregnancies')
plt.show()

skewness = round(data['Pregnancies'].skew(), 2) # 偏度
kurtosis = round(data['Pregnancies'].kurt(), 2) # 峰度
print(f"偏度(Skewness): {skewness}, 峰度(Kurtosis): {kurtosis}")

# 分布圖
sns.histplot(data['Pregnancies'], kde=True)
plt.title(f"Skewness: {skewness}, Kurtosis: {kurtosis}")
plt.show()
"""
# 移除離群值
# 將Outlier去掉，避免對Model造成影響。
print("Shape Of The Before Ouliers: ",data.shape)
n=1.5
# IQR = Q3-Q1
IQR = np.percentile(data['Pregnancies'],75) - np.percentile(data['Pregnancies'],25)
# outlier = Q3 + n*IQR 
transform_data = data[data['Pregnancies'] < np.percentile(data['Pregnancies'],75)+n*IQR]
# outlier = Q1 - n*IQR 
transform_data = transform_data[transform_data['Pregnancies'] > np.percentile(data['Pregnancies'],25)-n*IQR]
print ("Shape Of The After Ouliers: ",transform_data.shape)
print(transform_data.head())

# plt.boxplot(transform_data['Pregnancies'], showmeans=True)
# plt.title('Pregnancies')
# plt.show()


X = transform_data.drop('Outcome', axis=1)
Y = transform_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regr = LogisticRegression()
regr.fit(X_train, y_train)

# 預測
y_predictions = regr.predict(X_train)
test_accuracy = regr.score(X_test, y_test)
print(f'test accuracy : {test_accuracy}')
train_accuracy = regr.score(X_train, y_train)
print(f'train accuracy : {train_accuracy}')