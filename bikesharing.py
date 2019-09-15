import numpy as np
import pandas as pd
# getting the data set
dataset = pd.read_csv('hour.csv',sep = ',', error_bad_lines=False)
X = dataset.iloc[:, 2:14].values
y = dataset.iloc[:, 16].values




# dividing dataset so that some is used to train model and some is used to test how good it is
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
#making the feautures scaled so that some features don't dominate others

y_train = sc_y.fit_transform(y_train.reshape(-1,1))


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)


import statsmodels.formula.api as sm

#changing the variables we look for correlation in so we can predict the dependent variable
#without an excessive amount of independent variables
X = np.append(arr = np.ones((17379,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,4,5,6,8,9,10,11,12]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[1,2,4,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[2,4,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


