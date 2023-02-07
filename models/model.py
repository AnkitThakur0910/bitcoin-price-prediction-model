import pandas as pd
import numpy as np
from sklearn.svm import SVR
import pickle
import requests
import json



dataset = pd.read_csv('BTC-2017min (1).csv')
dataset.head()

dataset['date'] = pd.to_datetime(dataset['date'])
dataset['Hour'] = dataset['date'].dt.hour
dataset['Min'] = dataset['date'].dt.minute
dataset['Sec'] = dataset['date'].dt.second
dataset['Day'] = dataset['date'].dt.day
dataset['Month'] = dataset['date'].dt.month
dataset['date'] = dataset['date'].astype('string')
dataset['date'] = dataset['date'].str.replace('/', '')
dataset['date'] = dataset['date'].str.replace(':', '')
dataset['date'] = dataset['date'].str.replace(' ', '')
dataset['date'] = dataset['date'].str.replace('-', '')
dataset['date'] = dataset['date'].astype(float)

X = dataset['date'].values


"""## Feature Scaling"""

# we have to apply feature scaling on the svr model because model neglect the small values when there is higher value 
# on one side
# transform method convert in two d array
from sklearn.preprocessing import StandardScaler
Y = dataset.iloc[:,3].values
scalerX = StandardScaler()
scalerY = StandardScaler()
X = scalerX.fit_transform(X.reshape(-1,1))
Y = scalerY.fit_transform(Y.reshape(-1,1))


"""## Training the SVR model on the whole dataset"""

# we will use rbf kernel
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf",C = 1e3, gamma = 1e-8, epsilon = 0.1)
regressor.fit(X.reshape(-1,1),Y.ravel())

pickle.dump(regressor,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

"""## Predicting a new result"""

# inverse transform transform the 2d array of feature scaling into original 2d array
print(scalerY.inverse_transform(regressor.predict(scalerX.transform([[1.33768382e+13]])).reshape(-1,1)))
def prediction(val):
    return scalerY.inverse_transform(regressor.predict(scalerX.transform([[val]])).reshape(-1,1))

"""## Visualising the SVR results"""

