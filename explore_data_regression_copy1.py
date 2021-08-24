# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:26:48 2021

@author: dl7le
"""

import pandas as pd


# modelling algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors    import KNeighborsRegressor
from sklearn.ensemble     import AdaBoostRegressor
from sklearn.ensemble     import RandomForestRegressor
from xgboost import XGBRegressor
# tools
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import ParameterGrid

from datetime import datetime

# visualisation
import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel(r'c:/Users/dl7le/aigo/input_data_2021-01-18.xlsx', sheet_name='FALVITTABS30',
                    usecols=['Unnamed: 0', 'matryca_widocznosci', 'matryca_gazetki', 'Stock Units','MS%_Stock_Units', 'Stock Cover in Months',
                            'Purchase Price', 'MS%_Sell-out_Units','Retail Price','MS%_Sell-out_Value',
                            'ND Stock', 'Numeric Handling', 'ND Selling', 'MKT_Stock_Cover_in_Months',
                            'WD Stock', 'Środki Trade Marketing', 'Gross Sales', 'Net sales', 
                            'volume', 'mean_temperature', 'sell-in', 'sell-out'])

data = data.rename(columns={'Unnamed: 0': 'date'})
data.set_index('date')
data.columns
data.shape
data.info()

data.isnull().sum()
data.describe()

# check correlation graph

corr = data.corr()
fig, ax = plt.subplots(figsize=(25,25))
sns.heatmap(data=corr, square=True, annot=True, cbar=True, fmt='.2f')


sns.pairplot(data)

# check distribution 

sns.kdeplot(data['matryca_widocznosci'], shade=True, color='r')
plt.hist(data['matryca_gazetki'], bins=25)
sns.distplot(data['sell-out'])

# analyze feature by feature, create hypotesis try to find evidence

plt.figure(figsize=(12,8))
plt.plot(data['date'], data['sell-out'])
plt.xlabel('date')
plt.ylabel('sell-out')
plt.show()


plt.scatter(data['Stock Units'], data['sell-out']) # retail price -0.83 
plt.scatter(data['Retail Price'], data['sell-out']) 
plt.scatter(data['Numeric Handling'], data['sell-out']) 
plt.scatter(data['ND Selling'], data['sell-out'])  # numeric handling 0.73
plt.scatter(data['mean_temperature'], data['sell-out'])
plt.scatter(data['sell-in'], data['sell-out'])





# classifications 
sns.boxplot(data['matryca_widocznosci'], data['sell-out'])
sns.boxplot(data['matryca_gazetki'], data['sell-out'])


# splitting data into features X, and y labels

X = data[['date','Numeric Handling','matryca_widocznosci', 'Retail Price',
          'mean_temperature','sell-in', 'sell-out']]
X = X.set_index('date')


y = data[['date','sell-out']]
y = y.set_index('date')

# splitting data into train and test data validation data

X_train = X.shift(1).loc[:'2019-12-01', ['Numeric Handling','matryca_widocznosci', 'Retail Price','mean_temperature','sell-in', 'sell-out']].dropna().values
y_train = y.loc['2018-02-01':'2019-12-01', 'sell-out'].values.reshape(-1,1)




X_valid = X.shift(1).loc['2020-01-01':'2020-05-01', ['Numeric Handling','matryca_widocznosci', 'Retail Price', 'mean_temperature','sell-in', 'sell-out']].values
y_valid = y.loc['2020-01-01':'2020-05-01', 'sell-out'].values.reshape(-1,1)


X_test = X.shift(1).loc['2020-06-01':'2020-10-01', ['Numeric Handling','matryca_widocznosci', 'Retail Price', 'mean_temperature','sell-in', 'sell-out']].values
y_test = y.loc['2020-06-01':'2020-10-01', 'sell-out'].values.reshape(-1,1)



# scaling values

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_valid = sc.fit_transform(X_valid)



# test scores algorithms

scores = []
models = ['Linear Regression', 'Lasso Regression', 'AdaBoost Regression', 
          'Ridge Regression', 
          'KNeighbours Regression', 'Linear Regression by Temp', 'RandomForestRegressor', 'XGBRegressor']



# Linear regression
lr = LinearRegression(n_jobs=1, normalize=False)
lr.fit(X_train , y_train)
y_pred = lr.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)



scores.append(mape)
print('Linear Regression MAE: {0:.2f}'.format(mape))
 
 
# Lasso
lasso = Lasso(alpha=1e-05, normalize=False)
lasso.fit(X_train , y_train)
y_pred = lasso.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
 
scores.append(mape)
print('Lasso Regression MAE: {0:.2f}'.format(mape))
 
 
# Adaboost classifier
adaboost = AdaBoostRegressor(n_estimators=1000, learning_rate=43, loss='square')
adaboost.fit(X_train , y_train)
y_pred = adaboost.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
 
scores.append(mape)
print('AdaBoost Regression MAE: {0:.2f}'.format(mape))
 
# Ridge
ridge = Ridge(alpha=1e-05, normalize=False)
ridge.fit(X_train , y_train)
y_pred = ridge.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
 
scores.append(mape)
print('Ridge Regression MAE: {0:.2f}'.format(mape))
 
 
# K-Neighbours
kneighbours = KNeighborsRegressor(weights='distance', n_neighbors=8, leaf_size=25)
kneighbours.fit(X_train , y_train)
y_pred = kneighbours.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
 
scores.append(mape)
print('K-Neighbours Regression MAE: {0:.2f}'.format(mape))

# Linear Regression Temp
lr_temp = LinearRegression(normalize=False, fit_intercept=True, n_jobs=1)
lr_temp.fit(X_train[:, -3].reshape(-1,1), y_train)
y_pred = lr_temp.predict(X_test[:, -3].reshape(-1,1))
mape = mean_absolute_percentage_error(y_test, y_pred)

scores.append(mape)
print('Linear Regression Temp MAE: {0:.2f}'.format(mape))


rfr = RandomForestRegressor(n_estimators=30, max_depth=4, max_features=2, min_samples_split = 2, random_state=42)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)

scores.append(mape)
 
print('RandomForestRegressor MAE: {0:.2f}'.format(mape))


xgbr = XGBRegressor(alpha=0.0001, booster='gblinear',feature_selector= 'cyclic', updater='coord_descent').fit(X_train, y_train)
y_pred = xgbr.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)

scores.append(mape)

print('XGBRegressor MAE: {0:.2f}'.format(mape))


ranking = pd.DataFrame({'Algorithms' : models , 'Mean absolute percentage error' : scores})
ranking = ranking.sort_values(by='Mean absolute percentage error' ,ascending=True)
ranking
 
sns.barplot(x='Mean absolute percentage error' , y='Algorithms' , data=ranking)
plt.show()



# Validation 



def test_validation(model, grid_dist):

    grid = ParameterGrid(grid_dist)
    
    results = []
    params = []
    
    for i in grid:
        _model = model(**i).fit(X_train, y_train)
        
        y_pred = _model.predict(X_valid)
        mape = mean_absolute_percentage_error(y_valid, y_pred)
        results.append(mape)
        params.append(i)
        
    
    results_params = list(zip(results, params))
    minimum = min(map(lambda i: i[0], results_params))
    
    for i in results_params:
        if i[0] == minimum:
            print(i)
            break



# Lasso 

param_dist_Lasso= {'alpha': [1e-5, 1e-4,1e-3, 1e-2, 1e-1 ,0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    'normalize': [False]
                    }

test_validation(Lasso, param_dist_Lasso)


# AdaBoostRegressor

param_dist_ABR = {'learning_rate': range(1, 50),
                  'loss': ['linear', 'square', 'exponential'],
                  'n_estimators': [20, 30, 40, 100]
                  }

test_validation(AdaBoostRegressor, param_dist_ABR)

 
# Ridge 

param_dist_Ridge = {'alpha': [1e-5, 1e-4,1e-3, 1e-2, 1e-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    'normalize': [False]
                    }


test_validation(Ridge, param_dist_Ridge)

# KNeighborsRegressor

param_dist_KNR = {'n_neighbors': range(2,10),
                  'weights': ['uniform', 'distance'],
                  'leaf_size': range(25, 35),
                  }

test_validation(KNeighborsRegressor, param_dist_KNR)



# RandomForestRegressor

param_dist_rfr = {'max_depth': range(1,11),
              'max_features': range(1,6),
              'min_samples_split': range(2,11),
              'n_estimators': [30,40,100,1000],
              'random_state': [42]}

time_start = datetime.now()
test_validation(RandomForestRegressor, param_dist_rfr)
time_end = datetime.now()

print('time calculate: ', time_end - time_start)




param_dist_xgbr = {'booster': ['gblinear'],
                    'updater': ['shotgun', 'coord_descent'],
                   'feature_selector': ['cyclic', 'shuffle'],
                   
                   'alpha': [1e-5, 1e-4,1e-3, 1e-2, 1e-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                   }


test_validation(XGBRegressor, param_dist_xgbr)


