import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from scipy.stats import bernoulli, truncnorm
from sklearn.cluster import KMeans


# Given Parameters

P_r = 80
P_max = 270

# CSV files & filtering

## Energy forecast file

file_path_Energy = '.\Stochastic_Approach\Scenarios\Energy_forecast.csv'

df = pd.read_csv(file_path_Energy)

filtered_df_Energy = df[(df['forecast_da'] > 0) & (df['forecast_rt'] > 0)]

E_0_values = df['forecast_da'].tolist()

## Day-Ahead price

directory_path_da = '.\Stochastic_Approach\Scenarios\모의 실시간시장 가격\하루전'

files = os.listdir(directory_path_da)
csv_files = [file for file in files if file.endswith('일간.csv')]

def process_da_file(file_path):
    df = pd.read_csv(file_path)
    data = df.loc[3:27, df.columns[2]]  
    return data.tolist()

day_ahead_prices_daily = []


for csv_file in csv_files:
    file_path = os.path.join(directory_path_da, csv_file)
    processed_data = process_da_file(file_path)
    day_ahead_prices_daily.append(processed_data)

day_ahead_prices = [item for sublist in day_ahead_prices_daily for item in sublist]

## Real-Time price

directory_path_rt = '.\Stochastic_Approach\Scenarios\모의 실시간시장 가격\실시간 임시'

files_rt = os.listdir(directory_path_rt)

csv_files_rt = [file for file in files_rt if file.endswith('.csv')]

def process_rt_file(file_path):

    df = pd.read_csv(file_path)
    data = df.iloc[3:99, 2]  
    reshaped_data = data.values.reshape(-1, 4).mean(axis=1)
    return reshaped_data

real_time_prices_daily = []

for xlsx_file in csv_files_rt:
    file_path = os.path.join(directory_path_rt, xlsx_file)
    processed_data = process_rt_file(file_path)
    real_time_prices_daily.append(processed_data)

real_time_prices = [item for sublist in real_time_prices_daily for item in sublist]

print(len(day_ahead_prices), len(real_time_prices), len(E_0_values))

## Dataframe for TGMM

date_range = pd.date_range(start='2024-03-01', periods=6432, freq='D')[:6432]
data = pd.DataFrame({
    'date': date_range,
    'day_ahead_price': day_ahead_prices,
    'real_time_price': real_time_prices,
    'E_0_value': E_0_values
})

# Delta_E distribution

filtered_df_Energy['delta'] = filtered_df_Energy['forecast_rt'] / filtered_df_Energy['forecast_da']

delta_values = filtered_df_Energy['delta']

std_E = delta_values.std()  

lower_E, upper_E = 0.7, 1.3

a_E, b_E = (lower_E - 1) / std_E, (upper_E - 1) / std_E

Energy_dist = truncnorm(a_E, b_E, loc=1, scale=std_E)


# Q_c distribution

std_c = 0.5

lower_c, upper_c = -1, 1

a_c, b_c = (lower_c) / std_c, (upper_c) / std_c

Q_c_truncnorm_dist = truncnorm(a_c, b_c, loc=0, scale=std_c)

p = 0.05

def f_X(x):
    if x == 0:
        return 0.95  
    elif lower_c <= x <= upper_c:
        return p * Q_c_truncnorm_dist.pdf(x)
    else:
        return 0 

Q_c_x_values = np.linspace(-1.5, 1.5, 500)  
Q_c_f_X_values = np.array([f_X(x) for x in Q_c_x_values])


# Price Distributions

LB_price = -P_r
UB_price = P_max
LB_energy = -540
UB_energy = 18000
num_bins = 10

bin_edges_price = np.linspace(LB_price, UB_price, num_bins + 1)
bin_edges_energy = np.linspace(LB_energy, UB_energy, num_bins + 1)

def merge_bins(data, conditioning_var, bin_edges, min_data_per_bin=30):
    """
    Merge bins with fewer than min_data_per_bin data points.
    """
    data['bin'] = pd.cut(data[conditioning_var], bins=bin_edges, include_lowest=True)
    bin_counts = data['bin'].value_counts().sort_index()
    
    # Identify bins to merge
    bins_to_merge = bin_counts[bin_counts < min_data_per_bin].index.tolist()
    
    for bin_interval in bins_to_merge:
        idx = bin_edges.tolist().index(bin_interval.left)
        if idx > 0:
            # Merge with the previous bin
            new_left = bin_edges[idx - 1]
            new_right = bin_interval.right
            bin_edges[idx - 1] = new_left
            bin_edges = np.delete(bin_edges, idx)
            print(f"Merged bin {bin_interval} with previous bin.")
        elif idx < len(bin_edges) - 1:
            # Merge with the next bin
            new_left = bin_interval.left
            new_right = bin_edges[idx + 1]
            bin_edges[idx] = new_right
            bin_edges = np.delete(bin_edges, idx + 1)
            print(f"Merged bin {bin_interval} with next bin.")
    data.drop('bin', axis=1, inplace=True)
    return bin_edges
    
bin_edges_price = merge_bins(data, 'day_ahead_price', bin_edges_price, min_data_per_bin=30)
print("Adjusted Bin Edges (Price):", bin_edges_price)

bin_edges_energy = merge_bins(data, 'E_0_value', bin_edges_energy, min_data_per_bin=30)
print("Adjusted Bin Edges (Energy):", bin_edges_energy)

def truncated_gaussian_pdf(x, mean, std, lower, upper):

    a, b = (lower - mean) / std, (upper - mean) / std
    pdf = truncnorm.pdf(x, a, b, loc=mean, scale=std)
    pdf = np.where(np.isfinite(pdf), pdf, 0)
    return pdf

def initialize_parameters(K, data):

    kmeans = KMeans(n_clusters=K, random_state=42).fit(data.reshape(-1, 1))
    means = kmeans.cluster_centers_.flatten()
    stds = np.std(data) * np.ones(K)
    weights = np.ones(K) / K
    return weights, means, stds

def e_step(data, weights, means, stds, lower, upper):

    K = len(weights)
    responsibilities = np.zeros((len(data), K))
    
    for k in range(K):
        responsibilities[:, k] = weights[k] * truncated_gaussian_pdf(data, means[k], stds[k], lower, upper)
    
    responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
    responsibilities_sum[responsibilities_sum == 0] = 1e-10
    
    responsibilities /= responsibilities_sum
    return responsibilities

def m_step(data, responsibilities, lower, upper, min_std=1e-3):

    K = responsibilities.shape[1]
    Nk = responsibilities.sum(axis=0)
    weights = Nk / len(data)
    means = np.zeros(K)
    stds = np.zeros(K)
    
    for k in range(K):
        if Nk[k] == 0:
            means[k] = np.random.choice(data)
            stds[k] = np.std(data)
            weights[k] = 1e-6  
            continue
        
        means[k] = np.sum(responsibilities[:, k] * data) / Nk[k]
        
        variance = np.sum(responsibilities[:, k] * (data - means[k])**2) / Nk[k]
        variance = max(variance, min_std**2)
        stds[k] = np.sqrt(variance)
    
    weights /= weights.sum()
    
    return weights, means, stds

def compute_log_likelihood(data, weights, means, stds, lower, upper, epsilon=1e-10):

    K = len(weights)
    likelihood = np.zeros((len(data), K))
    
    for k in range(K):
        likelihood[:, k] = weights[k] * truncated_gaussian_pdf(data, means[k], stds[k], lower, upper)
    
    total_likelihood = np.sum(likelihood, axis=1)
    total_likelihood = np.maximum(total_likelihood, epsilon)
    log_likelihood = np.sum(np.log(total_likelihood))
    return log_likelihood

def em_algorithm(data, K, lower, upper, max_iters=100, tol=1e-4, min_std=1e-3):

    weights, means, stds = initialize_parameters(K, data)
    log_likelihood_old = None
    
    for iteration in range(max_iters):
        try:
            # E-Step
            responsibilities = e_step(data, weights, means, stds, lower, upper)
            
            # M-Step
            weights, means, stds = m_step(data, responsibilities, lower, upper, min_std=min_std)
            
            log_likelihood = compute_log_likelihood(data, weights, means, stds, lower, upper)
            
            if log_likelihood_old is not None:
                if np.abs(log_likelihood - log_likelihood_old) < tol:
                    print(f'Converged at iteration {iteration}')
                    break
            log_likelihood_old = log_likelihood
            
            if iteration % 10 == 0 or iteration == max_iters - 1:
                print(f'Iteration {iteration}: Log Likelihood = {log_likelihood:.4f}')
        except Exception as e:
            print(f'Error at iteration {iteration}: {e}')
            break
    
    return weights, means, stds, responsibilities

def plot_tgmm(data, weights, means, stds, lower, upper, title='TGMM Fit'):

    x = np.linspace(lower, upper, 1000)
    pdf = np.zeros_like(x)
    
    for w, m, s in zip(weights, means, stds):
        pdf += w * truncated_gaussian_pdf(x, m, s, lower, upper)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Data Histogram')
    plt.plot(x, pdf, 'k', linewidth=2, label='TGMM Fit')
    plt.title(title)
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def sample_from_tgmm(weights, means, stds, lower, upper, num_samples=1):

    components = np.random.choice(len(weights), size=num_samples, p=weights)
    samples = []
    for comp in components:
        a, b = (lower - means[comp]) / stds[comp], (upper - means[comp]) / stds[comp]
        sample = truncnorm.rvs(a, b, loc=means[comp], scale=stds[comp])
        samples.append(sample)
    return np.array(samples)


def inspect_bins(data, conditioning_var, target_var, num_bins):

    data['bin'] = pd.qcut(data[conditioning_var], q=num_bins, duplicates='drop')
    for bin_interval in data['bin'].unique():
        subset = data[data['bin'] == bin_interval][target_var]
        count = len(subset)
        min_val = subset.min()
        max_val = subset.max()
        print(f'Bin: {bin_interval}, Count: {count}, Min: {min_val:.2f}, Max: {max_val:.2f}')
    data.drop('bin', axis=1, inplace=True)

def plot_histogram(data, target_var, LB, UB):

    plt.figure(figsize=(10, 6))
    plt.hist(data[target_var], bins=100, range=(LB, UB), density=True, alpha=0.6, color='g')
    plt.title(f'Histogram of {target_var}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

def train_conditional_tgmm(data, conditioning_var, target_var, bin_edges, K=3, min_data_per_bin=30):

    data['bin'] = pd.cut(data[conditioning_var], bins=bin_edges, include_lowest=True)
    
    unique_bins = data['bin'].dropna().unique()
    
    tgmm_params = {}
    
    for bin_interval in unique_bins:
        subset = data[data['bin'] == bin_interval][target_var].values
        count = len(subset)
        if count < min_data_per_bin:
            print(f'Bin {bin_interval} has only {count} data points. Skipping.')
            continue
        print(f'Training TGMM for bin: {bin_interval} with {count} data points.')
        try:
            weights, means, stds, responsibilities = em_algorithm(subset, K, LB_price, UB_price)
            tgmm_params[bin_interval] = {
                'weights': weights,
                'means': means,
                'stds': stds
            }
            ##plot_tgmm(subset, weights, means, stds, LB_price, UB_price, title=f'TGMM Fit for {conditioning_var} in {bin_interval}')
        except Exception as e:
            print(f'Failed to train TGMM for bin {bin_interval}: {e}')
    
    data.drop('bin', axis=1, inplace=True)
    return tgmm_params, bin_edges


print("Inspecting bins for Model 1: day_ahead_price | E_0_value")
inspect_bins(data, 'E_0_value', 'day_ahead_price', num_bins)
##plot_histogram(data, 'day_ahead_price', LB_price, UB_price)

print("\nInspecting bins for Model 2: real_time_price | day_ahead_price")
inspect_bins(data, 'day_ahead_price', 'real_time_price', num_bins)
##plot_histogram(data, 'real_time_price', LB_price, UB_price)

print("\nTraining Model 1: day_ahead_price | E_0_value")
tgmm_model1_params, model1_bin_edges = train_conditional_tgmm(
    data=data,
    conditioning_var='E_0_value',
    target_var='day_ahead_price',
    bin_edges=bin_edges_energy,  
    min_data_per_bin=30
)

print("\nTraining Model 2: real_time_price | day_ahead_price")
tgmm_model2_params, model2_bin_edges = train_conditional_tgmm(
    data=data,
    conditioning_var='day_ahead_price',
    target_var='real_time_price',
    bin_edges=bin_edges_price,  
    K=3,
    min_data_per_bin=30
)


def find_bin(value, bin_edges, tgmm_params):

    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= value <= bin_edges[i + 1]:
            bin_interval = pd.Interval(left=bin_edges[i], right=bin_edges[i + 1], closed='right')
            if bin_interval in tgmm_params:
                return bin_interval
            else:
                break  
    closest_bin = min(tgmm_params.keys(), key=lambda x: min(abs(value - x.left), abs(value - x.right)))
    return closest_bin


def sample_day_ahead_price(E0_value, tgmm_params, bin_edges, num_samples=1):

    bin_interval = find_bin(E0_value, bin_edges, tgmm_params)
    if bin_interval is None:
        raise ValueError("E0_value could not be assigned to any bin.")
    params = tgmm_params.get(bin_interval)
    if params is None:
        bin_interval = find_bin(E0_value, bin_edges, tgmm_params)
        params = tgmm_params.get(bin_interval)
        if params is None:
            raise ValueError(f"No TGMM parameters found for the bin: {bin_interval}.")
    samples = sample_from_tgmm(params['weights'], params['means'], params['stds'], LB_price, UB_price, num_samples)
    return samples

new_E0_value = 2000 

try:
    sampled_day_ahead = sample_day_ahead_price(new_E0_value, tgmm_model1_params, model1_bin_edges, num_samples=100)
    print(f'\nSampled day_ahead_price values for E0_value={new_E0_value}:')
    print(sampled_day_ahead)
except ValueError as e:
    print(e)

def sample_real_time_price(DA_price, tgmm_params, bin_edges, num_samples=1):

    bin_interval = find_bin(DA_price, bin_edges, tgmm_params)
    if bin_interval is None:
        raise ValueError("day_ahead_price could not be assigned to any bin.")
    params = tgmm_params.get(bin_interval)
    if params is None:
        bin_interval = find_bin(DA_price, bin_edges, tgmm_params)
        params = tgmm_params.get(bin_interval)
        if params is None:
            raise ValueError(f"No TGMM parameters found for the bin: {bin_interval}.")
    samples = sample_from_tgmm(params['weights'], params['means'], params['stds'], LB_price, UB_price, num_samples)
    return samples

new_DA_price = -70  

try:
    sampled_real_time = sample_real_time_price(new_DA_price, tgmm_model2_params, model2_bin_edges, num_samples=1000)
    print(f'\nSampled real_time_price values for day_ahead_price={new_DA_price}:')
    print(sampled_real_time)
except ValueError as e:
    print(e)
    
# plotting distributions

## P_da
"""
x_P_da = np.linspace(LB_price, UB_price, 1000)
y_P_da = sampled_day_ahead


plt.figure(figsize=(10, 6))
plt.plot(x_P_da, y_P_da, label='Truncated Normal Distribution (delta_E)')
plt.title('Truncated Normal Distribution Fit to Delta Values')
plt.xlabel('Delta')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

## P_rt

x_P_rt = np.linspace(LB_price, UB_price, 1000)
y_P_rt = sampled_real_time

plt.figure(figsize=(10, 6))
plt.plot(x_P_rt, y_P_rt, label='Truncated Normal Distribution (delta_E)')
plt.title('Truncated Normal Distribution Fit to Delta Values')
plt.xlabel('Delta')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
"""
## Energy forecast

if __name__ == '__main__':
    x_E = np.linspace(lower_E, upper_E, 1000)
    y_E = Energy_dist.pdf(x_E)

    plt.figure(figsize=(10, 6))
    plt.plot(x_E, y_E, label='Truncated Normal Distribution (delta_E)')
    plt.title('Truncated Normal Distribution Fit to Delta Values')
    plt.xlabel('Delta')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    ## Q_c

    x_c = np.linspace(lower_c, upper_c, 1000)
    y_c = Q_c_truncnorm_dist.pdf(x_c)

    plt.plot(Q_c_x_values, Q_c_f_X_values, label="PDF of X")
    plt.axvline(0, color='r', linestyle='--', label="Probability mass at X=0")
    plt.title("PDF of X = Q_c")
    plt.xlabel("x")
    plt.ylabel("f_X(x)")
    plt.legend()
    plt.grid()
    plt.show()