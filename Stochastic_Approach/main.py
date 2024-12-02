import pyomo.environ as pyo
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

solver='gurobi'
SOLVER=pyo.SolverFactory(solver)

assert SOLVER.available(), f"Solver {solver} is available."

print("Hello")