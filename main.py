import pyomo.environ as pyo
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

solver='gurobi'
SOLVER=pyo.SolverFactory(solver)

SOLVER.options['Threads'] = 1  # Limit to 1 thread
SOLVER.options['TimeLimit'] = 2  # Set a time limit of 600 seconds

assert SOLVER.available(), f"Solver {solver} is available."

# CONSTANTS

# Day-Ahead price

directory_path_da = '.\모의 실시간시장 가격\하루전'

files = os.listdir(directory_path_da)
csv_files = [file for file in files if file.endswith('.csv')]

def process_file(file_path):
    df = pd.read_csv(file_path)
    data = df.loc[3:27, df.columns[2]]  
    return data.tolist()

day_ahead_prices = []


for csv_file in csv_files:
    file_path = os.path.join(directory_path_da, csv_file)
    processed_data = process_file(file_path)
    day_ahead_prices.append(processed_data)


# Real-Time price

directory_path_rt = '.\모의 실시간시장 가격\실시간 확정'

files_rt = os.listdir(directory_path_rt)

csv_files_rt = [file for file in files_rt if file.endswith('.csv')]

def process_file(file_path):

    df = pd.read_csv(file_path)
    data = df.iloc[3:99, 2]  
    reshaped_data = data.values.reshape(-1, 4).mean(axis=1)
    return reshaped_data

real_time_prices = []

for xlsx_file in csv_files_rt:
    file_path = os.path.join(directory_path_rt, xlsx_file)
    processed_data = process_file(file_path)
    real_time_prices.append(processed_data)


# E_0

file_path_E_0 = 'jeju_forecast.csv' 

df_E_0 = pd.read_csv(file_path_E_0)

df_E_0['timestamp'] = pd.to_datetime(df_E_0['timestamp'])

df_E_0['hour'] = df_E_0['timestamp'].dt.hour

average_forecast = df_E_0.groupby('hour')['gen_forecast'].mean()

E_0_mean = []

for i in average_forecast:
    E_0_mean.append(i)


# Other Params

C = 560
S = 168
S_min = 16.8
S_max = 151.2
P_r = 80
M = 10**6
T = 24
v = 0.95


# Scenario generation

def generate_scenario(n):
    np.random.seed(n)
    P_da=[]
    P_rt=[]
    E_0=[]
    E_1=[]
    U=[]
    I=np.random.binomial(1, 0.05, size = T)
    Un=np.random.uniform(0, 1, T)
    for t in range(T):
        P_da.append(day_ahead_prices[n][t])
        P_rt.append(day_ahead_prices[n][t])
        E_0.append(E_0_mean[t])
        E_1.append(E_0[t] * np.random.uniform(0.9, 1.1))
        U.append(I[t]*Un[t])
    scenario = [P_da, P_rt, E_0, E_1, U]
    return scenario

scenarios = []

for n in range(len(real_time_prices)):
    s = generate_scenario(n)
    scenarios.append(s) 
    


class deterministic_setting_1(pyo.ConcreteModel):
    def __init__ (self, n):
        super().__init__("Deterministic_Setting1")
        
        self.solved = False        
        self.n = n        
        self.scenario = scenarios[n]        
        self.P_da = self.scenario[0]        
        self.P_rt = self.scenario[1]        
        self.E_0 = self.scenario[2]        
        self.E_1 = self.scenario[3]        
        self.U = self.scenario[4]
        
        self.b_da_values = []
        self.b_rt_values = []
        self.q_da_values = []
        self.q_rt_values = []
        self.S_values = []
        
    def build_model(self):
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.ESSTIME = pyo.RangeSet(0, T)
        
        model.b_da = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.b_rt = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.g = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.d = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.u = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.ESSTIME, bounds=(S_min, S_max), domain=pyo.Reals) 
        
        model.y_da = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_rt = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_S = pyo.Var(model.TIME, domain=pyo.Binary)
        
        model.Q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        
        # Linearization Real Variables
        model.m1_V = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m2_V = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.m1_E = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.m2_E = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m3_E = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m4_E = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.m5_E = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.m1_Im = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.m2_Im = pyo.Var(model.TIME, domain=pyo.Reals)
        model.S1_V = pyo.Var(model.TIME, domain=pyo.Reals)
        model.S1_E = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.S1_Im = pyo.Var(model.TIME, domain=pyo.Reals)

        # Linearization Binary Variables
        model.n1_V = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_V = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n3_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n4_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n5_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n3_F = pyo.Var(model.TIME, domain=pyo.Binary)
        
        #부가정산금
        model.f_max = pyo.Var(model.TIME, domain=pyo.Reals)

        #Constraints

        if hasattr(model, 'constrs'):
            model.del_component('constrs')
            model.del_component('constrs_index')
        
        model.constrs = pyo.ConstraintList()
    
        for t in range(T):
            # q_da, q_rt constraint
            model.constrs.add(model.q_da[t] <= 1.1*self.E_0[t])
            model.constrs.add(model.q_rt[t] <= 1.1*self.E_1[t] + S_max - S_min)
            
            # Demand response
            model.constrs.add(model.b_da[t] - self.P_da[t] <= M * (1-model.y_da[t]))
            model.constrs.add(model.b_da[t] - self.P_da[t] <= M * model.y_da[t])
            model.constrs.add(model.Q_da[t] == model.y_da[t] * model.q_da[t])
        
            model.constrs.add(model.b_rt[t] - self.P_rt[t] <= M * (1-model.y_rt[t]))
            model.constrs.add(model.b_rt[t] - self.P_rt[t] <= M * model.y_rt[t])
            model.constrs.add(model.Q_rt[t] == model.y_rt[t]) 
            
            model.constrs.add(model.Q_c[t] == model.Q_rt[t]*self.U[t])
 
            # b_rt <= b_da
            model.constrs.add(model.b_rt[t] <= model.b_da[t])
            
            # ESS operation
            model.constrs.add(model.S[t+1] == model.S[t] + v*model.c[t] - (model.d[t])/v)
            model.constrs.add(model.u[t] == model.g[t] + (model.d[t])/v - v*model.c[t])
            model.constrs.add(model.g[t] <= self.E_1[t])
            model.constrs.add(model.c[t] <= model.g[t])
            model.constrs.add(model.S[0] == S_max)
            model.constrs.add(model.S[24] == S_max)
            
            
            #f_V constraint
            model.constrs.add(model.S1_V[t] == model.b_rt[t] * model.m1_V[t] - model.Q_da[t] * self.P_da[t] - model.m1_V[t] * self.P_rt[t] + self.P_rt[t] * model.Q_da[t])
            model.constrs.add(model.m1_V[t] <= model.u[t])
            model.constrs.add(model.m1_V[t] <= model.Q_c[t])
            model.constrs.add(model.m1_V[t] >= model.u[t] - M * (1 - model.n1_V[t]))
            model.constrs.add(model.m1_V[t] >= model.Q_c[t] - M * model.n1_V[t])
        
            model.constrs.add(model.m2_V[t] >= model.S1_V[t])
            model.constrs.add(model.m2_V[t] >= 0)
            model.constrs.add(model.m2_V[t] <= model.S1_V[t] + M * (1 - model.n2_V[t]))
            model.constrs.add(model.m2_V[t] <= M * model.n2_V[t])

            # f_E linearization constraints
            model.constrs.add(model.S1_E[t] == self.P_rt[t] - model.b_da[t])

            model.constrs.add(model.m1_E[t] <= model.Q_da[t])
            model.constrs.add(model.m1_E[t] <= model.q_rt[t])
            model.constrs.add(model.m1_E[t] >= model.Q_da[t] - M * (1 - model.n1_E[t]))
            model.constrs.add(model.m1_E[t] >= model.q_rt[t] - M * model.n1_E[t])

            model.constrs.add(model.m2_E[t] <= model.u[t])
            model.constrs.add(model.m2_E[t] <= model.Q_da[t])
            model.constrs.add(model.m2_E[t] >= model.u[t] - M * (1 - model.n2_E[t]))
            model.constrs.add(model.m2_E[t] >= model.Q_da[t] - M * model.n2_E[t])

            model.constrs.add(model.m3_E[t] >= model.m2_E[t])
            model.constrs.add(model.m3_E[t] >= model.Q_c[t])
            model.constrs.add(model.m3_E[t] <= model.m2_E[t] + M * (1 - model.n3_E[t]))
            model.constrs.add(model.m3_E[t] <= model.Q_c[t] + M * model.n3_E[t])

            model.constrs.add(model.m4_E[t] <= model.m3_E[t])
            model.constrs.add(model.m4_E[t] <= model.q_rt[t])
            model.constrs.add(model.m4_E[t] >= model.m3_E[t] - M * (1 - model.n4_E[t]))
            model.constrs.add(model.m4_E[t] >= model.q_rt[t] - M * model.n4_E[t])

            model.constrs.add(model.m5_E[t] >= (model.m1_E[t] - model.m4_E[t])*model.S1_E[t])
            model.constrs.add(model.m5_E[t] >= 0)
            model.constrs.add(model.m5_E[t] <= (model.m1_E[t] - model.m4_E[t])*model.S1_E[t] + M * (1 - model.n5_E[t]))
            model.constrs.add(model.m5_E[t] <= M * model.n5_E[t])
        
            model.constrs.add(model.f_max[t] >= model.m2_V[t])
            model.constrs.add(model.f_max[t] >= model.m5_E[t]) 
            model.constrs.add(model.f_max[t] >= 0)
            model.constrs.add(model.f_max[t] <= model.m2_V[t] + M*(1-model.n1_F[t]))
            model.constrs.add(model.f_max[t] <= model.m5_E[t] + M*(1-model.n2_F[t]))
            model.constrs.add(model.f_max[t] <= M*(1-model.n3_F[t]))
            model.constrs.add(model.n1_F[t]+model.n2_F[t]+model.n3_F[t]==1)        
        
            # f_Im linearization constraints
            model.constrs.add(model.S1_Im[t] == (model.u[t] - model.Q_c[t]) - 0.12 * C)

            model.constrs.add(model.m1_Im[t] >= model.S1_Im[t])
            model.constrs.add(model.m1_Im[t] >= 0)
            model.constrs.add(model.m1_Im[t] <= model.S1_Im[t] + M * (1 - model.n1_Im[t]))
            model.constrs.add(model.m1_Im[t] <= M * model.n1_Im[t])

            model.constrs.add(model.m2_Im[t] >= self.P_rt[t] - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] >= - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] <= self.P_rt[t] - model.b_rt[t] + M * (1 - model.n2_Im[t]))
            model.constrs.add(model.m2_Im[t] <= - model.b_rt[t] + M * model.n2_Im[t])
        
        # Objective Function
            
        model.objective = pyo.Objective(expr=sum(self.P_da[t] * model.Q_da[t] + self.P_rt[t] * (model.u[t] - model.Q_da[t]) + model.f_max[t] + (- model.m1_Im[t] * model.m2_Im[t]) + model.u[t] * P_r for t in model.TIME), sense=pyo.maximize)

    def solve(self):
        self.build_model()
        print(f"problem{n} solving")
        SOLVER.solve(self)
        self.solved = True
        
    def report(self):
        if not self.solved:
            self.solve()
            self.solved = True            
        print(f"\noptimal value = {pyo.value(self.objective)}")
            
    def objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
            
        return pyo.value(self.objective)
    
    def solve_with_fixed_vars(self, b_da_values, b_rt_values, q_da_values, q_rt_values, g_values, c_values, d_values, u_values):
        
        self.build_model()   
        model = self.model()
        
        for t in range(T):
            model.b_da[t].fix(b_da_values[t])
            model.b_rt[t].fix(b_rt_values[t])
            model.q_da[t].fix(q_da_values[t])
            model.q_rt[t].fix(q_rt_values[t])
            model.g[t].fix(g_values[t])
            model.c[t].fix(c_values[t])
            model.d[t].fix(d_values[t])
            model.u[t].fix(u_values[t])
                                    
        self.solve()
        
        for t in range(T):
            model.b_da[t].unfix()
            model.b_rt[t].unfix()
            model.q_da[t].unfix()
            model.q_rt[t].unfix()
    
        
        return self.objective_value()


class deterministic_setting_2(pyo.ConcreteModel):
    def __init__ (self, n):
        super().__init__("Deterministic_Setting2")
        
        self.solved = False        
        self.n = n        
        self.scenario = scenarios[n]        
        self.P_da = self.scenario[0]        
        self.P_rt = self.scenario[1]        
        self.E_0 = self.scenario[2]        
        self.E_1 = self.scenario[3]        
        self.U = self.scenario[4]
        
        self.b_da_values = []
        self.b_rt_values = []
        self.q_da_values = []
        self.u_values = []
        self.g_values = []
        self.c_values = []
        self.d_values = []
        self.z_values = []
        self.S_values = []
        
    def build_model(self):
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.ESSTIME = pyo.RangeSet(0, T)
        
        model.b_da = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.b_rt = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.g = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.d = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.u = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.z = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.ESSTIME, bounds=(S_min, S_max), domain=pyo.Reals) 
        
        model.y_da = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_rt = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_S = pyo.Var(model.TIME, domain=pyo.Binary)
        
        model.Q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        
        # Linearization Real Variables
        model.m1_V = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m1_E = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.m1_Im = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.m2_Im = pyo.Var(model.TIME, domain=pyo.Reals)
        model.S1_V = pyo.Var(model.TIME, domain=pyo.Reals)
        model.S1_E = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.S1_Im = pyo.Var(model.TIME, domain=pyo.Reals)

        # Linearization Binary Variables
        model.n1_V = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n3_F = pyo.Var(model.TIME, domain=pyo.Binary)
        
        #부가정산금
        model.f_max = pyo.Var(model.TIME, domain=pyo.Reals)



        #Constraints

        if hasattr(model, 'constrs'):
            model.del_component('constrs')
            model.del_component('constrs_index')
        
        model.constrs = pyo.ConstraintList()
    
        for t in range(T):
            # q_da, q_rt constraint
            model.constrs.add(model.q_da[t] <= 1.1*self.E_0[t])
            
            # Demand Response
            model.constrs.add(model.b_da[t] - self.P_da[t] <= M * (1-model.y_da[t]))
            model.constrs.add(model.b_da[t] - self.P_da[t] <= M * model.y_da[t])
            model.constrs.add(model.Q_da[t] == model.y_da[t] * model.q_da[t])
        
            model.constrs.add(model.b_rt[t] - self.P_rt[t] <= M * (1-model.y_rt[t]))
            model.constrs.add(model.b_rt[t] - self.P_rt[t] <= M * model.y_rt[t])
            model.constrs.add(model.z[t] == model.y_rt[t] * model.u[t]) 
            
            model.constrs.add(model.Q_c[t] == model.z[t]*self.U[t])
 
            # b_rt <= b_da
            model.constrs.add(model.b_rt[t] <= model.b_da[t])
            
            # ESS operation
            model.constrs.add(model.S[t+1] == model.S[t] + v*model.c[t] - (model.d[t])/v)
            model.constrs.add(model.z[t] == model.g[t] + (model.d[t])/v - v*model.c[t])
            model.constrs.add(model.g[t] <= self.E_1[t])
            model.constrs.add(model.c[t] <= model.g[t])
            model.constrs.add(model.S[0] == S_max)
            model.constrs.add(model.S[24] == S_max)
            
            #f_V constraint
            model.constrs.add(model.S1_V[t] == model.b_rt[t] * model.z[t] - model.Q_da[t] * self.P_da[t] - model.z[t] * self.P_rt[t] + self.P_rt[t] * model.Q_da[t])
        
            model.constrs.add(model.m1_V[t] >= model.S1_V[t])
            model.constrs.add(model.m1_V[t] >= 0)
            model.constrs.add(model.m1_V[t] <= model.S1_V[t] + M * (1 - model.n1_V[t]))
            model.constrs.add(model.m1_V[t] <= M * model.n1_V[t])

            # f_E linearization constraints
            model.constrs.add(model.S1_E[t] == self.P_rt[t] - model.b_da[t])

            model.constrs.add(model.m1_E[t] >= (model.Q_da[t] - model.z[t])*model.S1_E[t])
            model.constrs.add(model.m1_E[t] >= 0)
            model.constrs.add(model.m1_E[t] <= (model.Q_da[t] - model.z[t])*model.S1_E[t] + M * (1 - model.n1_E[t]))
            model.constrs.add(model.m1_E[t] <= M * model.n1_E[t])
            
            model.constrs.add(model.f_max[t] >= model.m1_V[t])
            model.constrs.add(model.f_max[t] >= model.m1_E[t]) 
            model.constrs.add(model.f_max[t] >= 0)
            model.constrs.add(model.f_max[t] <= model.m1_V[t] + M*(1-model.n1_F[t]))
            model.constrs.add(model.f_max[t] <= model.m1_E[t] + M*(1-model.n2_F[t]))
            model.constrs.add(model.f_max[t] <= M*(1-model.n3_F[t]))
            model.constrs.add(model.n1_F[t]+model.n2_F[t]+model.n3_F[t]==1)
        
            # f_Im linearization constraints
            model.constrs.add(model.S1_Im[t] == (model.z[t] - model.Q_c[t]) - 0.12 * C)

            model.constrs.add(model.m1_Im[t] >= model.S1_Im[t])
            model.constrs.add(model.m1_Im[t] >= 0)
            model.constrs.add(model.m1_Im[t] <= model.S1_Im[t] + M * (1 - model.n1_Im[t]))
            model.constrs.add(model.m1_Im[t] <= M * model.n1_Im[t])

            model.constrs.add(model.m2_Im[t] >= self.P_rt[t] - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] >= - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] <= self.P_rt[t] - model.b_rt[t] + M * (1 - model.n2_Im[t]))
            model.constrs.add(model.m2_Im[t] <= - model.b_rt[t] + M * model.n2_Im[t])
    
        # Objective Function
        model.objective = pyo.Objective(expr=sum(self.P_da[t] * model.Q_da[t] + self.P_rt[t] * (model.z[t] - model.Q_da[t]) + model.f_max[t] + (- model.m1_Im[t] * model.m2_Im[t]) + model.z[t] * P_r for t in model.TIME), sense=pyo.maximize)
    
    def solve(self):
        self.build_model()
        SOLVER.solve(self)
        print(f"problem{n} solved.")
        self.solved = True
        
    def report(self):
        if not self.solved:
            self.solve()
            self.solved = True
            
        print(f"\noptimal value = {pyo.value(self.objective)}")
        

        if not self.solved:
            self.solve()
            self.solved = True
        for t in range(T):
            self.S_values.append(pyo.value(self.S[t]))
        return self.S_values

    def optimal_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        for t in range(T):
            self.b_da_values.append(pyo.value(self.b_da[t]))
            self.b_rt_values.append(pyo.value(self.b_rt[t]))
            self.q_da_values.append(pyo.value(self.q_da[t]))
            self.u_values.append(pyo.value(self.u[t]))
            self.g_values.append(pyo.value(self.g[t]))
            self.c_values.append(pyo.value(self.c[t]))
            self.d_values.append(pyo.value(self.d[t]))
            self.z_values.append(pyo.value(self.z[t]))
            self.S_values.append(pyo.value(self.S[t]))     
      
    def objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
            
        return pyo.value(self.objective)

r = range(len(scenarios)-69)
Tr = range(T)


b_da_list = []
b_rt_list = []
q_da_list = []
u_list = []
g_list = []
c_list = []
d_list = []
z_list = []

for n in r:
    det = deterministic_setting_2(n)
    det.optimal_solutions()
    b_da_list.append(det.b_da_values)
    b_rt_list.append(det.b_rt_values)
    q_da_list.append(det.q_da_values)
    u_list.append(det.u_values)
    g_list.append(det.g_values)
    c_list.append(det.c_values)
    d_list.append(det.d_values)
    z_list.append(det.z_values)



"""
for n in r:
    plt.plot(Tr, q_da_list[n])
    
plt.xlabel('Time')
plt.ylabel('q_da values')
plt.title('q_da')

plt.ylim(0, 300)

plt.legend()

plt.show()

"""

d1_obj = []
d2_obj = []
difference = []


for n in r:
    d1 = deterministic_setting_1(n)
    d1_1 = deterministic_setting_1(n)
    d1_o = d1.objective_value()
    d2 = d1_1.solve_with_fixed_vars(
        b_da_list[n], b_rt_list[n], q_da_list[n], u_list[n], g_list[n], c_list[n], d_list[n], z_list[n]
    )
    d1_obj.append(d1_o)
    d2_obj.append(d2)
    difference.append(abs(d1_o-d2))

plt.plot(r, d1_obj, label='Original')
plt.plot(r, d2_obj, label='Approximation')
plt.plot(r, difference, label='Abs difference')

plt.xlabel('Scenario Index')
plt.ylabel('Values')
plt.title('Comparison of Deterministic Settings')

plt.ylim(0, 1000000)

plt.legend()

plt.show()

print(difference)


"""
for n in r:
    d1 =deterministic_setting_1(n)
    d2 = deterministic_setting_2(n)
    d1_obj.append(d1.objective_value())
    d2_obj.append(d2.objective_value())
    difference.append(abs(d1.objective_value()-d2.objective_value()))

plt.plot(r, d1_obj, label='Original')
plt.plot(r, d2_obj, label='Approximation')
plt.plot(r, difference, label='Abs difference')

plt.xlabel('Scenario Index')
plt.ylabel('Values')
plt.title('Comparison of Deterministic Settings')

plt.ylim(0, 1000000)

plt.legend()

plt.show()
"""



