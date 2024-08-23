import pandas as pd
import os

directory_path = '.\모의 실시간시장 가격\하루전'

files = os.listdir(directory_path)
csv_files = [file for file in files if file.endswith('.csv')]

def process_file(file_path):
    df = pd.read_csv(file_path)
    data = df.loc[3:27, df.columns[2]]  
    return data.tolist()

processed_dataframes = []


for csv_file in csv_files:
    file_path = os.path.join(directory_path, csv_file)
    processed_data = process_file(file_path)
    processed_dataframes.append(processed_data)
    
print(processed_dataframes[1])







