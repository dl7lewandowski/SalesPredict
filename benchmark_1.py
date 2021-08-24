# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:00:34 2021

@author: dl7le
"""

# modelling algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors    import KNeighborsRegressor
from sklearn.ensemble     import AdaBoostRegressor
from sklearn.ensemble     import RandomForestRegressor

# tools
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import ParameterGrid


import pandas as pd
import numpy as np
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



# splitting data into features X, and y labels


X = data[['date','Stock Units','matryca_gazetki', 'Purchase Price', 'mean_temperature','sell-in', 
          'sell-out']]
X = X.set_index('date')


y = data[['date','sell-out']]
y = y.set_index('date')



# splitting data into train and test data validation data

X_train = X.loc[:'2019-12-01', 'sell-out'].values.mean(axis=0).reshape(1,-1)

# Wygenerować tyle watosci co w y_train X_train i policzyć miary dokładnoci

X_train = np.full(shape=(24), fill_value=X_train.copy(), dtype=np.int64).reshape(-1,1)
X_train.shape


y_train = y.loc[:'2019-12-01', 'sell-out'].values
y_train.shape


X_valid = X.loc[:'2019-12-01',  'sell-out'].values.mean(axis=0).reshape(1,-1)
X_valid = np.full(shape=5, fill_value=X_valid.copy(), dtype=np.int64).reshape(-1,1)
X_valid.shape

y_valid = y.loc['2020-01-01':'2020-05-01', 'sell-out'].values
y_valid.shape

X_test = X.loc[:'2020-05-01', 'sell-out'].values.mean(axis=0).reshape(1,-1)
X_test = np.full(shape=5, fill_value=X_test.copy(), dtype=np.int64).reshape(-1,1)

y_test = y.loc['2020-06-01':'2020-10-01', 'sell-out'].values


# benchmark na podstawie wartosci sredniej


mape = mean_absolute_percentage_error(y_train, X_train)
print('Data train, mean absolue percente error: {}'.format(mape))

mape = mean_absolute_percentage_error(y_valid, X_valid)
print('Data validation, mean absolue percente error: {}'.format(mape))

mape = mean_absolute_percentage_error(y_test, X_test)
print('Data test, mean absolue percente error: {}'.format(mape))





scores = []

scores.append(mean_absolute_percentage_error(y_train, X_train))
scores.append(mean_absolute_percentage_error(y_valid, X_valid))
scores.append(mean_absolute_percentage_error(y_test, X_test))

data_type = ['mae: data train', 'mae: data validation', 'mae: data test']
ranking = pd.DataFrame({'data_type' : data_type, 'Mean absolute percentage error' : scores})
ranking = ranking.sort_values(by='Mean absolute percentage error' ,ascending=True)
ranking

sns.barplot(x='Mean absolute percentage error' , y='data_type' , data=ranking)
plt.show()









