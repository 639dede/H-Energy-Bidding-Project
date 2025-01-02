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

# SDDiP Model

class SDDiPModel:

    def __init__(self, T=24, num_scenarios=10, alpha=0.95):

        self.T = T
        self.M = num_scenarios
        self.alpha = alpha

        self.LB = -np.inf
        self.UB = np.inf

        self.iteration = 0

        self.forward_solutions = [
            [None for _ in range(self.M)] for _ in range(self.T + 1)
        ]

        self.psi = [[] for _ in range(self.T + 1)]
        
        self.data = {}

    def sample_scenarios(self):
 
        scenarios = []
        for _ in range(self.M):
            xi = [random.random() for _ in range(self.T)]
            scenarios.append(xi)
        return scenarios

    def solve_forward_subproblem(self, t, x_prev, xi_t):

        solution = {
            "x": 0.0,    
            "y": 0.0,   
            "z": 0.0,  
            "theta": 0.0 
        }
        return solution

    def compute_stage_cost(self, solution, xi_t):

        return solution["x"] + solution["y"] + solution["z"]

    def forward_pass(self, scenarios):

        scenario_costs = np.zeros(self.M)
        
        for k in range(self.M):
            x_prev = None  
            for t in range(1, self.T + 1):
                xi_t = scenarios[k][t-1]
                
                solution = self.solve_forward_subproblem(t, x_prev, xi_t)
                self.forward_solutions[t][k] = solution

                scenario_costs[k] += self.compute_stage_cost(solution, xi_t)

                x_prev = solution["x"]

        return scenario_costs

    def update_upper_bound(self, scenario_costs):

        M = len(scenario_costs)
        mu_hat = np.mean(scenario_costs)
        sigma_hat = np.std(scenario_costs, ddof=1) 

        from math import sqrt
        z_alpha_half = 1.96 
        
        self.UB = mu_hat + z_alpha_half * (sigma_hat / np.sqrt(M))

    def solve_backward_subproblem(self, t, x_prev, xi_t):

        nu = 0.0
        mu = 0.0
        pi = 0.0
        return nu, mu, pi

    def add_benders_cut(self, t, cut_coeffs):

        self.psi[t].append(cut_coeffs)

    def backward_pass(self, scenarios):

        for t in range(self.T, 1, -1):
            for k in range(self.M):
                xi_t = scenarios[k][t-1]

                x_prev = self.forward_solutions[t-1][k]["x"] if t > 1 else 0.0

                nu, mu, pi = self.solve_backward_subproblem(t, x_prev, xi_t)

                self.add_benders_cut(t, (nu, mu, pi))

    def update_lower_bound(self):

        self.LB = max(self.LB, 0.0 + self.iteration)  # dummy logic

    def stopping_criterion(self, tol=1e-3, max_iter=50):

        if self.iteration >= max_iter:
            return True
        gap = self.UB - self.LB
        return (gap < tol)

    def run_sddip(self):

        
        while not self.stopping_criterion():
            self.iteration += 1
            print(f"\n=== Iteration {self.iteration} ===")

            scenarios = self.sample_scenarios()

            scenario_costs = self.forward_pass(scenarios)

            self.update_upper_bound(scenario_costs)
            print(f"  UB updated to: {self.UB:.4f}")

            self.backward_pass(scenarios)

            self.update_lower_bound()
            print(f"  LB updated to: {self.LB:.4f}")

        print("\nSDDiP complete.")
        print(f"Final LB = {self.LB:.4f}, UB = {self.UB:.4f}, gap = {self.UB - self.LB:.4f}")


if __name__ == "__main__":
    sddip = SDDiPModel(T=24, num_scenarios=5, alpha=0.95)
    sddip.run_sddip()

    
