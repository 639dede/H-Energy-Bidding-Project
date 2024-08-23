import pandas as pd
import os

directory_path = '.\모의 실시간시장 가격\실시간 확정'

files = os.listdir(directory_path)

xlsx_files = [file for file in files if file.endswith('.csv')]

def process_file(file_path):

    df = pd.read_csv(file_path)
    data = df.iloc[3:99, 2]  
    reshaped_data = data.values.reshape(-1, 4).mean(axis=1)
    return reshaped_data

processed_dataframes = []


for xlsx_file in xlsx_files:
    file_path = os.path.join(directory_path, xlsx_file)
    processed_data = process_file(file_path)
    processed_dataframes.append(processed_data)

print(processed_dataframes[1])
    