import pandas as pd

def E_0_mean_values(file_path):

    file_path = 'jeju_forecast.csv' 
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    average_forecast = df.groupby('hour')['gen_forecast'].mean()
    E_0_mean = []
    for i in average_forecast:
        E_0_mean.append(i)
    return E_0_mean

