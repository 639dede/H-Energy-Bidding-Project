import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
import random
import time

scenarios_path = os.path.join(os.path.dirname(__file__), "Scenarios")
sys.path.append(scenarios_path)

from Scenarios import Setting1_A_scenario

Energy_dist = Setting1_A_scenario.Energy_dist
Q_c_truncnorm_dist = Setting1_A_scenario.Q_c_truncnorm_dist
Q_c_x_values = Setting1_A_scenario.Q_c_x_values
Q_c_f_X_values = Setting1_A_scenario.Q_c_f_X_values
lower_E = Setting1_A_scenario.lower_E
upper_E = Setting1_A_scenario.upper_E
lower_c = Setting1_A_scenario.lower_c
upper_c = Setting1_A_scenario.upper_c


solver='gurobi'
SOLVER=pyo.SolverFactory(solver)

assert SOLVER.available(), f"Solver {solver} is available."

print("Hello")

    
# Plot the Energy forecast distribution
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

# Plot the Q_c distribution
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