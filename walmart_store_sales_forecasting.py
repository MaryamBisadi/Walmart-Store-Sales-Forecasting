import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt

from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Importing the datset
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
store=pd.read_csv('stores.csv')
feature=pd.read_csv('features.csv')

merge_df=pd.merge(train,feature, on=['Store','Date'], how='inner')

#plot weekly sales of store 1 based on date
merge_df['DateTimeObj']=[dt.strptime(x,'%Y-%m-%d') for x in list(merge_df['Date'])]
merge_df['DateTimeObj'].head()
plt.plot(merge_df[(merge_df.Store==1)].DateTimeObj, merge_df[(merge_df.Store==1)].Weekly_Sales, 'ro')
plt.show()

weeklysales=merge_df.groupby(['Store','Date'])['Weekly_Sales'].apply(lambda x:np.sum(x))
weeklyscale=weeklysales.reset_index()

walmartstore=pd.merge(weeklyscale, feature, on=['Store', 'Date'], how='inner')

#plot sum of weekly sales of different departments for store 1 based on date
walmartstore['DateTimeObj'] = [dt.strptime(x, '%Y-%m-%d') for x in list(walmartstore['Date'])]
walmartstore['DateTimeObj'].head()
plt.plot(walmartstore[(walmartstore.Store==1)].DateTimeObj, walmartstore[(walmartstore.Store==1)].Weekly_Sales, 'ro')
plt.show()

walmartstore['IsHolidayInt'] = [int(x) for x in list(walmartstore.IsHoliday)]
train_WM, test_WM = train_test_split(walmartstore, test_size=0.3,random_state=42)

walmartstore['Store'].unique()
Store_Dummies = pd.get_dummies(walmartstore.Store, prefix='Store').iloc[:,1:]

walmartstore = pd.concat([walmartstore, Store_Dummies], axis=1)
walmartstore.head()

# Splitting the dataset into Training set and Tes set
train_WM, test_WM = train_test_split(walmartstore, test_size=0.3,random_state=42)
XTrain = train_WM.iloc[:,list(range(3,5)) + list(range(10,13)) + list(range(14,walmartstore.shape[1]))]
YTrain = train_WM.Weekly_Sales
                                                    
XTest = test_WM.iloc[:,list(range(3,5)) + list(range(10,13)) + list(range(14,walmartstore.shape[1]))]
YTest=test_WM.Weekly_Sales
XTrain.head()
wmLinear = linear_model.LinearRegression(normalize=True)
wmLinear.fit(XTrain, YTrain)

#Performance on the test data sets
YHatTest = wmLinear.predict(XTest)
plt.plot(YTest, YHatTest,'ro')
plt.plot(YTest, YTest,'b-')
plt.show()

# calculate the accuray of the model by sum of Square and mean absolute prediction error
MAPE = np.mean(abs((YTest - YHatTest)/YTest))
MSSE = np.mean(np.square(YHatTest - YTest))

print("Linear Regression Mean Absolute Prediction Error:",MAPE)
print("Linear Regression Mean Squared Error", MSSE)

# Lasso
alphas = np.linspace(0, 30, 5)
testError = np.empty(5)

for i, alpha in enumerate(alphas):
    lasso = Lasso(alpha=alpha)
    lasso.fit(XTrain, YTrain)
    testError[i] = mean_squared_error(YTest, lasso.predict(XTest))

plt.plot(alphas, testError, 'r-')
plt.show()

#Ridge
alphas = np.linspace(0, 30, 5)
testError = np.empty(5)

for i, alpha in enumerate(alphas):
    ridge = Ridge(alpha=alpha)
    ridge.fit(XTrain, YTrain)
    testError[i] = mean_squared_error(YTest, ridge.predict(XTest))

plt.plot(alphas, testError, 'r-')
plt.show()

#ElasticNet
alphas = np.linspace(0, 30, 5)
testError = np.empty(5)

for i, alpha in enumerate(alphas):
    elasticnet = ElasticNet(alpha=alpha)
    elasticnet.fit(XTrain, YTrain)
    testError[i] = mean_squared_error(YTest, elasticnet.predict(XTest))

plt.plot(alphas, testError, 'r-')
plt.show()


# Random Forest Regression
n_estimators = np.linspace(5, 200, 5)
testError = np.empty(5)

for i, n_estimator in enumerate(n_estimators):
    randomforest = RandomForestRegressor(n_estimators = int(n_estimator), random_state = 0)
    randomforest.fit(XTrain, YTrain)
    testError[i] = mean_squared_error(YTest, randomforest.predict(XTest))

plt.plot(n_estimators, testError, 'r-')
plt.show()
