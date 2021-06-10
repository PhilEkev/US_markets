import pandas as pd
import numpy as np
import pmdarima as pm
import itertools

from statsmodels.tsa.stattools import acf
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima_model import ARIMA

def get_data(n=None):
    data = pd.read_csv('../raw_data/stock_prices_latest.csv', nrows=n)
    return data

def preprocess(data,s,period=3):
    
    # Choix de la période
    if period == 1:
        data = data[(data['data'] >= '2000-01-01') & (data['date'] <= '2006-12-31')]
    elif period == 2:
        data = data[(data['data'] >= '2007-01-01') & (data['date'] <= '2012-12-31')]
    else:
        data = data[(data['data'] >= '2013-01-01') & (data['date'] <= '2018-12-31')]
    
    # Filtre sur le symbol
    is_sym = data['symbol']==s
    df_sym = data[is_sym]
    
    # Tri par ordre croissant de date
    df_sym = df_sym.sort_values(by='date')
    
    #Création train/test
    df_len = int(len(df_sym)*0.99)
    train = df_sym[:df_len]
    test = df_sym[df_len:]
    
    # Récupération des dates test
    date_final = pd.Series(data.iloc[-len(test):]['date'])
    
    # Reset index et on garde que la liste close_adjusted
    train = train.reset_index().drop(columns='index')
    train = train["close_adjusted"]
    test = test.reset_index().drop(columns='index')
    test = test["close_adjusted"]
    
    return df_sym, train, test, date_final


def get_best_model(data,train,test):
    range_p = [0,1,2]
    range_d = [1]
    range_q = [0,1,2]
    grid = itertools.product(range_p, range_d, range_q)
    orders = []
    aics = []
    fold_idxs = []
    for (p,d,q) in grid:
        order = (p,d,q)
        folds = TimeSeriesSplit(n_splits=3)
        for fold_idx, (train_idx, test_idx) in enumerate(folds.split(data)):
            fold_idxs.append(fold_idx)
            y_train = train[train_idx]
            y_test = test[test_idx]
            model = ARIMA(y_train, order=order).fit()
            y_pred = model.forecast(len(y_test))[0]
            print(y_pred)
            orders.append(order)
            aics.append(model.aic)
            
            
    results = pd.DataFrame(list(zip(fold_idxs, orders, aics)), 
                    columns =['Fold', '(p,d,q)', 'AIC'])
    results = results.sort_values('AIC').groupby('(p,d,q)').mean()['AIC'].sort_values()
    best_order = results.index[0]
    return best_order

def forecast(model):
    (forecast, stderr, conf_int) = model.forecast(1, alpha=0.05)
    forecast = pd.Series(forecast, name='forecast')
    stderr = pd.Series(stderr)
    conf_int = pd.DataFrame(conf_int, columns=['low', 'high'])
    
    return forecast

    
def forecast_accuracy(y_pred: pd.Series, y_true: pd.Series) -> float:
    
    mape = np.mean(np.abs(y_pred - y_true)/np.abs(y_true))  # Mean Absolute Percentage Error
    me = np.mean(y_pred - y_true)             # ME
    mae = np.mean(np.abs(y_pred - y_true))    # MAE
    mpe = np.mean((y_pred - y_true)/y_true)   # MPE
    rmse = np.mean((y_pred - y_true)**2)**.5  # RMSE
    corr = np.corrcoef(y_pred, y_true)[0,1]   # Correlation between the Actual and the Forecast
    mins = np.amin(np.hstack([y_pred.values.reshape(-1,1), y_true.values.reshape(-1,1)]), axis=1)
    maxs = np.amax(np.hstack([y_pred.values.reshape(-1,1), y_true.values.reshape(-1,1)]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(y_pred-y_true, fft=False)[1]
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})