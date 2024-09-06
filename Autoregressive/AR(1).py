#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


data = pd.read_csv(".\Autoregressive\jeju_filtered_data.csv", index_col=0, parse_dates=True)

data['timestamp'] = pd.to_datetime(data['timestamp']).dt.date

data.set_index('timestamp')

m_t = data['forecast_rt']/data['forecast_da']

reshaped_array = m_t.values.reshape(230, 13)

# AR(1) Test

non_AR_1=[]

non_AR_num = 0

for i in range(len(reshaped_array)):
    ts = reshaped_array[i, :]

    # Fit an AR(1) model to the first time series
    model = sm.tsa.ARIMA(ts, order=(1, 0, 0)).fit()

    # Get the residuals
    residuals = model.resid

    # Perform the Ljung-Box test (lags=12 for 12 residual autocorrelations)
    ljung_box_result = acorr_ljungbox(residuals, lags=[12], return_df=True)

    p_value = ljung_box_result['lb_pvalue'].iloc[0]

    if p_value < 0.05:
        non_AR_1.append(i)
        non_AR_num+=1
        
print(non_AR_num)
