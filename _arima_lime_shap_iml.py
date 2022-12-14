# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
path ="/content/drive/MyDrive/csv/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv"
bitstamp = pd.read_csv(path)
# Dataset is now stored in a Pandas dataframe
bitstamp.describe

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import plotly.express as px
from itertools import product
import warnings
import statsmodels.api as sm
plt.style.use('seaborn-darkgrid')

#matplotlib inline

bitstamp.info()

# Converting the Timestamp column from string to datetime
bitstamp['Timestamp'] = [datetime.fromtimestamp(x) for x in bitstamp['Timestamp']]

bitstamp.head()

# Converting the Timestamp column from string to datetime
bitstamp['Timestamp'] = [datetime.fromtimestamp(x) for x in bitstamp['Timestamp']]

bitstamp.set_index("Timestamp").Weighted_Price.plot(figsize=(14,7), title="Bitcoin Weighted Price")

#calculating missing values in the dataset

missing_values = bitstamp.isnull().sum()
missing_per = (missing_values/bitstamp.shape[0])*100
missing_table = pd.concat([missing_values,missing_per], axis=1, ignore_index=True) 
missing_table.rename(columns={0:'Total Missing Values',1:'Missing %'}, inplace=True)
missing_table

#testing missing value methods on a subset

pd.set_option('display.max_rows', 1500)

a = bitstamp.set_index('Timestamp')

a = a['2019-11-01 00:15:00':'2019-11-01 02:24:00']

a['ffill'] = a['Weighted_Price'].fillna(method='ffill') # Imputation using ffill/pad
a['bfill'] = a['Weighted_Price'].fillna(method='bfill') # Imputation using bfill/pad
a['interp'] = a['Weighted_Price'].interpolate()         # Imputation using interpolation

a

def fill_missing(df):
    ### function to impute missing values using interpolation ###
    df['Open'] = df['Open'].interpolate()
    df['Close'] = df['Close'].interpolate()
    df['Weighted_Price'] = df['Weighted_Price'].interpolate()

    df['Volume_(BTC)'] = df['Volume_(BTC)'].interpolate()
    df['Volume_(Currency)'] = df['Volume_(Currency)'].interpolate()
    df['High'] = df['High'].interpolate()
    df['Low'] = df['Low'].interpolate()

    print(df.head())
    print(df.isnull().sum())

fill_missing(bitstamp)

#created a copy 
bitstamp_non_indexed = bitstamp.copy()

bitstamp = bitstamp.set_index('Timestamp')
bitstamp.head()

ax = bitstamp['Weighted_Price'].plot(title='Bitcoin Prices', grid=True, figsize=(14,7))
ax.set_xlabel('Year')
ax.set_ylabel('Weighted Price')

ax.axvspan('2018-12-01','2019-01-31',color='red', alpha=0.3)
ax.axhspan(17500,20000, color='green',alpha=0.3)

#Zooming in

ax = bitstamp.loc['2017-10':'2019-03','Weighted_Price'].plot(marker='o', linestyle='-',figsize=(15,6), title="Oct-17 to March-19 Trend", grid=True)
ax.set_xlabel('Month')
ax.set_ylabel('Weighted_Price')

sns.kdeplot(bitstamp['Weighted_Price'], shade=True)

plt.figure(figsize=(15,12))
plt.suptitle('Lag Plots', fontsize=22)

plt.subplot(3,3,1)
pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=1) #minute lag
plt.title('1-Minute Lag')

plt.subplot(3,3,2)
pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=60) #hourley lag
plt.title('1-Hour Lag')

plt.subplot(3,3,3)
pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=1440) #Daily lag
plt.title('Daily Lag')

plt.subplot(3,3,4)
pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=10080) #weekly lag
plt.title('Weekly Lag')

plt.subplot(3,3,5)
pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=43200) #month lag
plt.title('1-Month Lag')

plt.legend()
plt.show()

hourly_data = bitstamp.resample('1H').mean()
hourly_data = hourly_data.reset_index()

hourly_data.head()

bitstamp_daily = bitstamp.resample("24H").mean() #daily resampling

import plotly.express as px

bitstamp_daily.reset_index(inplace=True)
fig = px.line(bitstamp_daily, x='Timestamp', y='Weighted_Price', title='Weighted Price with Range Slider and Selectors')
fig.update_layout(hovermode="x")

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(step="all")
            
        ])
    )
)
fig.show()

plot_ = bitstamp_daily.set_index("Timestamp")["2017-12"]

import plotly.graph_objects as go

fig = go.Figure(data=go.Candlestick(x= plot_.index,
                    open=plot_['Open'],
                    high=plot_['High'],
                    low=plot_['Low'],
                    close=plot_['Close']))
fig.show()

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fill_missing(bitstamp_daily)

plt.figure(figsize=(15,12))
series = bitstamp_daily.Weighted_Price
result = seasonal_decompose(series, model='additive',freq=1)
result.plot()

acf = plot_acf(series, lags=50, alpha=0.05)
plt.title("ACF for Weighted Price", size=20)
plt.show()

plot_pacf(series, lags=50, alpha=0.05, method='ols')
plt.title("PACF for Weighted Price", size=20)
plt.show()

stats, p, lags, critical_values = kpss(series, 'ct')

print(f'Test Statistics : {stats}')
print(f'p-value : {p}')
print(f'Critical Values : {critical_values}')

if p < 0.05:
    print('Series is not Stationary')
else:
    print('Series is Stationary')

def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    
    print (dfoutput)
    
    if p > 0.05:
        print('Series is not Stationary')
    else:
        print('Series is Stationary')

adf_test(series)

df = bitstamp_daily.set_index("Timestamp")

df.reset_index(drop=False, inplace=True)

lag_features = ["Open", "High", "Low", "Close","Volume_(BTC)"]
window1 = 3
window2 = 7
window3 = 30

df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

df_mean_3d = df_rolled_3d.mean().shift(1).reset_index()
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index()
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index()

df_std_3d = df_rolled_3d.std().shift(1).reset_index()
df_std_7d = df_rolled_7d.std().shift(1).reset_index()
df_std_30d = df_rolled_30d.std().shift(1).reset_index()

for feature in lag_features:
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
    
    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

df.fillna(df.mean(), inplace=True)

df.set_index("Timestamp", drop=False, inplace=True)
df.head()

df["month"] = df.Timestamp.dt.month
df["week"] = df.Timestamp.dt.week
df["day"] = df.Timestamp.dt.day
df["day_of_week"] = df.Timestamp.dt.dayofweek
df.head()

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')

from datetime import datetime

df_train = df[df.Timestamp < "2020"]
df_valid = df[df.Timestamp >= "2020"]

print('train shape :', df_train.shape)
print('validation shape :', df_valid.shape)

!pip install pmdarima

exogenous_features = ['Open_mean_lag3',
       'Open_mean_lag7', 'Open_mean_lag30', 'Open_std_lag3', 'Open_std_lag7',
       'Open_std_lag30', 'High_mean_lag3', 'High_mean_lag7', 'High_mean_lag30',
       'High_std_lag3', 'High_std_lag7', 'High_std_lag30', 'Low_mean_lag3',
       'Low_mean_lag7', 'Low_mean_lag30', 'Low_std_lag3', 'Low_std_lag7',
       'Low_std_lag30', 'Close_mean_lag3', 'Close_mean_lag7',
       'Close_mean_lag30', 'Close_std_lag3', 'Close_std_lag7',
       'Close_std_lag30', 'Volume_(BTC)_mean_lag3', 'Volume_(BTC)_mean_lag7',
       'Volume_(BTC)_mean_lag30', 'Volume_(BTC)_std_lag3',
       'Volume_(BTC)_std_lag7', 'Volume_(BTC)_std_lag30', 'month', 'week',
       'day', 'day_of_week']

from pmdarima.arima import auto_arima

len(df_valid)

model = auto_arima(df_train.Weighted_Price, exogenous=df_train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(df_train.Weighted_Price, exogenous=df_train[exogenous_features])

forecast = model.predict(n_periods= len(df_valid), exogenous=df_valid[exogenous_features])
df_valid["Forecast_ARIMAX"] = forecast

df_valid[["Weighted_Price", "Forecast_ARIMAX"]].plot(figsize=(14, 7))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.Weighted_Price, df_valid.Forecast_ARIMAX)))

print("\nMAE of Auto ARIMAX:", mean_absolute_error(df_valid.Weighted_Price, df_valid.Forecast_ARIMAX))

print("\nR2 of Auto ARIMAX:", r2_score(df_valid.Weighted_Price, df_valid.Forecast_ARIMAX))

model.plot_diagnostics()

pip install shap

pip install azureml-interpret

pip install dash-cytoscape

pip install lime

import pandas as pd #for manipulating data
import numpy as np #for manipulating data
import sklearn #for building models
from pmdarima.arima import auto_arima #for building models
import sklearn.ensemble #for building models
from sklearn.model_selection import train_test_split #for creating a hold-out sample
import lime #LIME package
import lime.lime_tabular #the type of LIIME analysis we???ll do
import time #some of the routines take a while so we monitor the time
import os #needed to use Environment Variables in Domino
import matplotlib.pyplot as plt #for custom graphs at the end
import seaborn as sns #for custom graphs at the end
# Importing the module for LimeTabularExplainer
import lime.lime_tabular

X_train, y_train = df_train[exogenous_features], df_train.Weighted_Price
X_test, y_test = df_valid[exogenous_features], df_valid.Weighted_Price

# if a feature has 10 or less unique values then treat it as categorical
categorical_features = np.argwhere(np.array([len(set(X_train.values[:,x]))
for x in range(X_train.values.shape[1])]) <= 10).flatten()
    # LIME has one explainer for all models
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values) 
feature_names=X_train.columns.values.tolist(),
class_names=['Weighted_Price'],
categorical_features=categorical_features,
mode='regression'

df["Timestamp"] = pd.to_datetime(df["Timestamp"], '%d/%m/%Y').dt.strftime('%Y%m%d').astype(float)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_score,
)
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
)

"""***Check Lime compatibility***"""

new_index = X_train.index.strftime("%s").astype("float")

X_train.index = new_index

dfnew_index = df_train.index.strftime("%s").astype("float")

df_train.index = dfnew_index

df_train["Timestamp"] = pd.to_datetime(df_train["Timestamp"], '%d/%m/%Y').dt.strftime('%Y%m%d').astype(float)

dfvnew_index = df_valid.index.strftime("%s").astype("float")

df_valid.index = dfvnew_index

#df_train[exogenous_features].values.reshape(-1, 1)

!pip install statsmodels

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from interpret import show
from interpret.blackbox import ShapKernel

#pca = PCA()
#model = auto_arima(df_train.Weighted_Price, exogenous=df_train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)

#blackbox_model = Pipeline([('pca', pca), ('model', model)])

#blackbox_model.fit(X_train, y_train.values.reshape(-1, 1))

#shap = ShapKernel(predict_fn=blackbox_model.predict, data=X_train)
#shap_local = shap.explain_local(X_test[:5], y_test[:5])

#show(shap_local)

pip install azureml-interpret

pip install dash

pip install dash-cytoscape

#df = bitstamp_daily.set_index("Timestamp")

import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show

df = df
df.columns = [
        'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)',         
        'Weighted_Price','Open_mean_lag3','Open_mean_lag7', 'Open_mean_lag30', 'Open_std_lag3', 'Open_std_lag7',
       'Open_std_lag30', 'High_mean_lag3', 'High_mean_lag7', 'High_mean_lag30',
       'High_std_lag3', 'High_std_lag7', 'High_std_lag30', 'Low_mean_lag3',
       'Low_mean_lag7', 'Low_mean_lag30', 'Low_std_lag3', 'Low_std_lag7',
       'Low_std_lag30', 'Close_mean_lag3', 'Close_mean_lag7',
       'Close_mean_lag30', 'Close_std_lag3', 'Close_std_lag7',
       'Close_std_lag30', 'Volume_(BTC)_mean_lag3', 'Volume_(BTC)_mean_lag7',
       'Volume_(BTC)_mean_lag30', 'Volume_(BTC)_std_lag3',
       'Volume_(BTC)_std_lag7', 'Volume_(BTC)_std_lag30', 'month', 'week',
       'day', 'day_of_week'
]

train_cols = df.columns[0:-1]
label = df.columns[-1]
X = df[train_cols]
y = df[label]

ebm = ExplainableBoostingRegressor()
ebm.fit(X, y)

ebm_global = ebm.explain_global()
show(ebm_global)

ebm_local = ebm.explain_local(X, y)
show(ebm_local)

import pandas as pd
from interpret.blackbox import ExplainableBoostingRegressor
from interpret import show

df = df
df.columns = [
        'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)',         
        'Weighted_Price','Open_mean_lag3','Open_mean_lag7', 'Open_mean_lag30', 'Open_std_lag3', 'Open_std_lag7',
       'Open_std_lag30', 'High_mean_lag3', 'High_mean_lag7', 'High_mean_lag30',
       'High_std_lag3', 'High_std_lag7', 'High_std_lag30', 'Low_mean_lag3',
       'Low_mean_lag7', 'Low_mean_lag30', 'Low_std_lag3', 'Low_std_lag7',
       'Low_std_lag30', 'Close_mean_lag3', 'Close_mean_lag7',
       'Close_mean_lag30', 'Close_std_lag3', 'Close_std_lag7',
       'Close_std_lag30', 'Volume_(BTC)_mean_lag3', 'Volume_(BTC)_mean_lag7',
       'Volume_(BTC)_mean_lag30', 'Volume_(BTC)_std_lag3',
       'Volume_(BTC)_std_lag7', 'Volume_(BTC)_std_lag30', 'month', 'week',
       'day', 'day_of_week'
]

train_cols = df.columns[0:-1]
label = df.columns[-1]
X = df[train_cols]
y = df[label]

ebm = ExplainableBoostingRegressor()
ebm.fit(X, y)

ebm_global = ebm.explain_global()
show(ebm_global)

ebm_local = ebm.explain_local(X, y)
show(ebm_local)

from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
import lime
import lime.lime_tabular

X = X_train.to_numpy()
y = y_train

ebm = ExplainableBoostingRegressor()
ebm.fit(X, y)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), feature_names=X_train.columns, verbose=True, mode='regression')
exp = explainer.explain_instance(X_test.iloc[0], 
     ebm.predict, num_features=34)
exp.as_pyplot_figure()

exp.show_in_notebook(show_table=True, show_all=False)

from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
import lime
import lime.lime_tabular

X = X_train.to_numpy()
y = y_train

ebm = ExplainableBoostingRegressor()
ebm.fit(X, y)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), feature_names=X_train.columns, verbose=True, mode='regression')
exp = explainer.explain_instance(X_test.iloc[0], 
     ebm.predict, num_features=34)
exp.as_pyplot_figure()

exp.show_in_notebook(show_table=True, show_all=False)

from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
import lime
import lime.lime_tabular

X = X_train.to_numpy()
y = y_train

ebm = ExplainableBoostingRegressor()
ebm.fit(X, y)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), feature_names=X_train.columns, verbose=True, mode='regression')
exp = explainer.explain_instance(X_train.iloc[0], 
     ebm.predict, num_features=34)
exp.as_pyplot_figure()

exp.show_in_notebook(show_table=True, show_all=False)

from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
import lime
import lime.lime_tabular

X = X_test.to_numpy()
y = y_test

ebm = ExplainableBoostingRegressor()
ebm.fit(X, y)

explainer = lime.lime_tabular.LimeTabularExplainer(X_test.to_numpy(), feature_names=X_test.columns, verbose=True, mode='regression')
exp = explainer.explain_instance(X_test.iloc[0], 
     ebm.predict, num_features=34)
exp.as_pyplot_figure()

exp.show_in_notebook(show_table=True, show_all=False)

def return_weights(exp):
    
    """Get weights from LIME explanation object"""
    
    exp_list = exp.as_map()[1]
    exp_list = sorted(exp_list, key=lambda x: x[0])
    exp_weight = [x[1] for x in exp_list]
    
    return exp_weight

weights = []

#Iterate over first 100 rows in feature matrix
for x in X_test.values[0:100]:
    
    #Get explanation
    exp = explainer.explain_instance(x, 
                                 ebm.predict, 
                                 num_features=34, 
                                labels=X_test.columns)
    
    #Get weights
    exp_weight = return_weights(exp)
    weights.append(exp_weight)
    
#Create DataFrame
lime_weights = pd.DataFrame(data=weights,columns=X_test.columns)

#Create DataFrame
lime_weights = pd.DataFrame(data=weights,columns=X_test.columns)

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,8))

#Get weights and feature values
feature_weigth = lime_weights['Open_mean_lag3']
feature_value = X_test['Open_mean_lag3'][0:100]

plt.scatter(x=feature_value ,y=feature_weigth)

plt.ylabel('LIME Weight',size=20)
plt.xlabel('Open_mean_lag3',size=20)

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,8))

#Get weights and feature values
feature_weigth = lime_weights['day_of_week']
feature_value = X_test['day_of_week'][0:100]

plt.scatter(x=feature_value ,y=feature_weigth)

plt.ylabel('LIME Weight',size=20)
plt.xlabel('day_of_week',size=20)

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,8))

#Get weights and feature values
feature_weigth = lime_weights['Open_std_lag30']
feature_value = X_test['Open_std_lag30'][0:100]

plt.scatter(x=feature_value ,y=feature_weigth)

plt.ylabel('LIME Weight',size=20)
plt.xlabel('Open_std_lag30',size=20)

#Get abs mean of LIME weights
abs_mean = lime_weights.abs().mean(axis=0)
abs_mean = pd.DataFrame(data={'feature':abs_mean.index, 'abs_mean':abs_mean})
abs_mean = abs_mean.sort_values('abs_mean')

#Plot abs mean
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,8))

y_ticks = range(len(abs_mean))
y_labels = abs_mean.feature
plt.barh(y=y_ticks,width=abs_mean.abs_mean)

plt.yticks(ticks=y_ticks,labels=y_labels,size= 15)
plt.title('')
plt.ylabel('')
plt.xlabel('Mean |Weight|',size=20)

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,6))

#Use same order as mean plot
y_ticks = range(len(abs_mean))
y_labels = abs_mean.feature

#plot scatterplot for each feature
for i,feature in enumerate(y_labels):
    
    feature_weigth = lime_weights[feature]
    feature_value = X_test[feature][0:100]
    
    plt.scatter(x=feature_weigth ,
                y=[i]*len(feature_weigth),
                c=feature_value,
                cmap='bwr',
                edgecolors='black',
               alpha=0.8)
    
plt.vlines(x=0,ymin=0,ymax=9,colors='black',linestyles="--")
plt.colorbar(label='Feature Value',ticks=[])

plt.yticks(ticks=y_ticks,labels=y_labels,size=15)
plt.xlabel('LIME Weight',size=20)

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb

from lime.lime_tabular import LimeTabularExplainer
import shap
shap.initjs()

#Get SHAP values
explainer = shap.Explainer(ebm)
shap_values = explainer(X_test[0:100])

#Replace SHAP values with LIME weights
shap_placeholder = explainer(X_test[0:100])
shap_placeholder.values = np.array(lime_weights)

import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show

#model = auto_arima(df_train.Weighted_Price, exogenous=df_train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)
#model.fit(df_train.Weighted_Price, exogenous=df_train[exogenous_features])

#forecast = model.predict(n_periods= len(df_valid), exogenous=df_valid[exogenous_features])
#df_valid["Forecast_ARIMAX"] = forecast

df = df
df.columns = [
        'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)',         
        'Weighted_Price','Open_mean_lag3','Open_mean_lag7', 'Open_mean_lag30', 'Open_std_lag3', 'Open_std_lag7',
       'Open_std_lag30', 'High_mean_lag3', 'High_mean_lag7', 'High_mean_lag30',
       'High_std_lag3', 'High_std_lag7', 'High_std_lag30', 'Low_mean_lag3',
       'Low_mean_lag7', 'Low_mean_lag30', 'Low_std_lag3', 'Low_std_lag7',
       'Low_std_lag30', 'Close_mean_lag3', 'Close_mean_lag7',
       'Close_mean_lag30', 'Close_std_lag3', 'Close_std_lag7',
       'Close_std_lag30', 'Volume_(BTC)_mean_lag3', 'Volume_(BTC)_mean_lag7',
       'Volume_(BTC)_mean_lag30', 'Volume_(BTC)_std_lag3',
       'Volume_(BTC)_std_lag7', 'Volume_(BTC)_std_lag30', 'month', 'week',
       'day', 'day_of_week'
]

train_cols = df_train.columns[0:-1]
label = df_train.columns[-1]
X = df_train[train_cols]
y = df_train[label]

ebm = ExplainableBoostingRegressor()
ebm.fit(X, y)

ebm_global = ebm.explain_global()
show(ebm_global)

ebm_local = ebm.explain_local(X, y)
show(ebm_local)

import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show

df = df
df.columns = [
        'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)',         
        'Weighted_Price','Open_mean_lag3','Open_mean_lag7', 'Open_mean_lag30', 'Open_std_lag3', 'Open_std_lag7',
       'Open_std_lag30', 'High_mean_lag3', 'High_mean_lag7', 'High_mean_lag30',
       'High_std_lag3', 'High_std_lag7', 'High_std_lag30', 'Low_mean_lag3',
       'Low_mean_lag7', 'Low_mean_lag30', 'Low_std_lag3', 'Low_std_lag7',
       'Low_std_lag30', 'Close_mean_lag3', 'Close_mean_lag7',
       'Close_mean_lag30', 'Close_std_lag3', 'Close_std_lag7',
       'Close_std_lag30', 'Volume_(BTC)_mean_lag3', 'Volume_(BTC)_mean_lag7',
       'Volume_(BTC)_mean_lag30', 'Volume_(BTC)_std_lag3',
       'Volume_(BTC)_std_lag7', 'Volume_(BTC)_std_lag30', 'month', 'week',
       'day', 'day_of_week'
]

train_cols = df_train.columns[0:-1]
label = df_train.columns[-1]
X = df_train[train_cols]
y = df_train[label]

ebm = ExplainableBoostingRegressor()
ebm.fit(X, y)

ebm_global = ebm.explain_global()
show(ebm_global)

ebm_local = ebm.explain_local(X, y)
show(ebm_local)

import lime
import lime.lime_tabular
# Instantiating the explainer object by passing in the training set, and the extracted features

model = ebm.fit(X, y)

explainer = lime.lime_tabular.LimeTabularExplainer(X.to_numpy(), feature_names=X.columns, verbose=True, mode='regression')
exp = explainer.explain_instance(X.iloc[0], 
     model.predict, num_features=34)
exp.as_pyplot_figure()
# plot feature importance

exp.show_in_notebook(show_table=True, show_all=False)

#model = auto_arima(df_train.Weighted_Price, exogenous=df_train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)
#model.fit(df_train.Weighted_Price, exogenous=df_train[exogenous_features])

#forecast = model.predict(n_periods= len(df_valid), exogenous=df_valid[exogenous_features])
#df_valid["Forecast_ARIMAX"] = forecast

import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show

X= df_train
y= df_valid.columns

ebm = ExplainableBoostingRegressor()
ebm.fit(X, y)

ebm_global = ebm.explain_global()
show(ebm_global)

ebm_local = ebm.explain_local(X, y)
show(ebm_local)

# fit a GAM model to the data
import interpret.glassbox
model_ebm = interpret.glassbox.ExplainableBoostingRegressor()
# compute SHAP values
import shap
def __init__(self, x=0, y=0):
    explainer = shap.KernelExplainer(model_ebm) 
    shap_values = explainer.shap_values(df[train_cols])

# fit a GAM model to the data
import interpret.glassbox
model_ebm = interpret.glassbox.ExplainableBoostingRegressor()
X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution
sample_ind = 18
model_ebm.fit(X, y)

# explain the GAM model with SHAP
explainer_ebm = shap.Explainer(model_ebm.predict, X100) # 100 instances for use as the background distribution)
shap_values_ebm = explainer_ebm(X)

# make a standard partial dependence plot with a single SHAP value overlaid
fig,ax = shap.partial_dependence_plot(
    "RM", model_ebm.predict, X, model_expected_value=True,
    feature_expected_value=True, show=False, ice=False,
    shap_values=shap_values_ebm[sample_ind:sample_ind+1,:]
)

df_train

from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
import lime
import lime.lime_tabular

#df_train[["Timestamp", "Weighted_Price"
X = df_train[Timestamp]
y = df_train[Weighted_Price]

ebm = ExplainableBoostingRegressor()
ebm.fit(X, y)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), feature_names=X_train.columns, verbose=True, mode='regression')
exp = explainer.explain_instance(X_test.iloc[0], 
     ebm.predict, num_features=34)
exp.as_pyplot_figure()

exp.show_in_notebook(show_table=True, show_all=False)
