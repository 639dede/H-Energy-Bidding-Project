#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


# In[2]:


data = pd.read_csv("C:/Users/gram/Desktop/데이터 분석/Sorted_m(t).csv", index_col=0, parse_dates=True)
data.head()


# In[4]:


m_t = data['m(t)']

# 이동 평균과 표준 편차 계산
window = 14 # 이동 평균 윈도우 크기
rolling_mean = m_t.rolling(window=window).mean()
rolling_std = m_t.rolling(window=window).std()

# 변화점 탐지
threshold = 3 # 표준 편차의 몇 배를 넘는지
change_points = np.where(np.abs(m_t - rolling_mean) > threshold*rolling_std)[0]

# 변화점 주변을 이상치로 간주하고 제거
outlier_indices = []
window_size = 1 # 이상치 제거 범위
for cp in change_points:
    outlier_indices.extend(range(max(0, cp - window_size), min(len(m_t), cp + window_size)))
    
m_t_filtered_cp = m_t.drop(m_t.index[outlier_indices])

# AR(1) 모델 피팅
model_cp = AutoReg(m_t_filtered_cp, lags=1)
model_cp_fitted = model_cp.fit()

# 모델 결과 및 잔차의 Ljung-Box 테스트
print(model_cp_fitted.summary())

residuals_cp = model_cp_fitted.resid
ljung_box_test_cp = sm.stats.acorr_ljungbox(residuals_cp, lags=[1], return_df=True)
print(ljung_box_test_cp)


# In[11]:


# Z-Score 방법
mean_m = data['m(t)'].mean()
std_m = data['m(t)'].std()
data['Z-Score'] = (data['m(t)'] - mean_m) / std_m

# Z-Score 이상치 탐지 (임계값 3)
z_score_outliers = data[np.abs(data['Z-Score']) > 3]

# IQR 방법
Q1 = data['m(t)'].quantile(0.25)
Q3 = data['m(t)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# IQR 이상치 탐지
iqr_outliers = data[(data['m(t)'] < lower_bound) | (data['m(t)'] > upper_bound)]

# 이상치 결합 및 중복 제거
outliers_combined = pd.concat([z_score_outliers, iqr_outliers]).drop_duplicates()

# 이상치 제거
cleaned_data = data[~data.index.isin(outliers_combined.index)]

# AR(1) 모델 피팅
ar_model = AutoReg(cleaned_data['m(t)'], lags=1).fit()

# 잔차 분석
residuals = ar_model.resid

# ACF, PACF 플롯 그리기
plt.figure(figsize=(12, 6))
plot_acf(residuals, lags=40)
plt.title('ACF of Residuals for AR(1) Model')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(residuals, lags=40, method='ywmle')
plt.title('PACF of Residuals for AR(1) Model')
plt.grid(True)
plt.show()

# 모델 요약 출력
print(ar_model.summary())

ljung_box_test_cp = sm.stats.acorr_ljungbox(residuals, lags=[1], return_df=True)
print(ljung_box_test_cp)

