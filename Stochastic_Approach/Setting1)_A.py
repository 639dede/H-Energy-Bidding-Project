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

# Generate Scenario

scenario_test = Setting1_A_scenario.Setting1_A_scenario(1)

print("Hello")

    
