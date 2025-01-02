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

E_0 = Setting1_A_scenario.E_0

solver='gurobi'
SOLVER=pyo.SolverFactory(solver)

assert SOLVER.available(), f"Solver {solver} is available."

# Generate Scenario

scenario_test = Setting1_A_scenario.Setting1_A_scenario(1, E_0)


if __name__ == '__main__':
    print(scenario_test.P_da)
    print(scenario_test.scenario())

    
