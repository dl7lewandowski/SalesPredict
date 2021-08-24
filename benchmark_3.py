# modelling algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors    import KNeighborsRegressor
from sklearn.ensemble     import AdaBoostRegressor
from sklearn.ensemble     import RandomForestRegressor

# tools
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
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


# benchmark z poprzedniego miesaca t-1, t+1

X_train = X.loc[:'2019-12-01', 'sell-out'].shift(1).dropna(axis=0).values.reshape(-1,1)
X_train.shape

y_train = y.loc['2018-02-01':'2019-12-01', 'sell-out'].values
y_train.shape

X_valid = X.shift(1).loc['2020-01-01':'2020-05-01', 'sell-out'].values.reshape(-1,1)
y_valid = y.loc['2020-01-01':'2020-05-01', 'sell-out'].values




X_test = X.shift(1).loc['2020-06-01':'2020-10-01', 'sell-out']
y_test = y.loc['2020-06-01':'2020-10-01', 'sell-out'].values


mean_absolute_percentage_error(y_train, X_train)


mean_absolute_percentage_error(y_valid, X_valid)


mean_absolute_percentage_error(y_test, X_test)

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



# naive model t-12, t+12 
scores = []
models = [LinearRegression, Lasso, Ridge, KNeighborsRegressor, AdaBoostRegressor, RandomForestRegressor]
for model in models:
    
    _model = model().fit(X_train, y_train)
    
    predictions = []
    
    for x in X_valid:
        y_pred = _model.predict([x])
        predictions.append(y_pred)
    score = mean_absolute_percentage_error(y_valid, predictions)
    print(str(model.__name__), 'Score: naive model', score)  
    scores.append(score)
    
    plt.plot(y_train)
    plt.plot([None for i in y_train] + [x for x in y_valid], color='red',) 
    plt.plot([None for i in y_train] + [x for x in predictions], color='gold')
    plt.title(str(model.__name__))
    plt.show()

# the same results

score = mean_absolute_percentage_error(y_train, X_train)

models = ['LinearRegression', 'Lasso', 'Ridge', 'KNeighborsRegressor', 'AdaBoostRegressor', 'RandomForestRegressor']
ranking = pd.DataFrame({'Algorithms' : models, 'Mean absolute percentage error' : scores})
ranking = ranking.sort_values(by='Mean absolute percentage error' ,ascending=True)
ranking
 
sns.barplot(x='Mean absolute percentage error' , y='Algorithms' , data=ranking)
plt.show()
