import numpy as np
import pyomo.environ as pyo

###############################################################################
# Stochastic Dual Dynamic Integer Programming (SDDiP) - 4-stage Skeleton (Pyomo)
# Revised version to fix BlockData attribute errors when creating constraints.
###############################################################################

T = 4
num_scenarios = 5      # Number of sampled scenarios per forward pass
max_iterations = 20    # Maximum number of forward-backward SDDiP iterations

# Generate “random” data for each stage
def generate_scenarios(num_scen):
    """
    Return a list of scenarios, each scenario is a list of (c_t, b_t) for t=1..T.
    """
    scenarios = []
    for _ in range(num_scen):
        scenario_t = []
        for t in range(T):
            c_t = np.random.uniform(1, 5)  # cost coefficient
            b_t = np.random.uniform(5, 10) # capacity or RHS
            scenario_t.append((c_t, b_t))
        scenarios.append(scenario_t)
    return scenarios

# Benders cuts storage: cuts[t] is a list of (pi, const) for stage t
# representing   ψ_t >= pi * x_t + const.
cuts = [[] for _ in range(T+1)]  # t=0 unused if you like, or t=1..T

# Global bounds
LB = -1e10
UB =  1e10

def solve_forward(scenario, cuts, warmstart=None):
    """
    Solve a 4-stage Pyomo model (one variable per stage) given a single scenario
    and the current Benders (future cost) cuts.
    
    scenario: list of (c_t, b_t) for t=0..T-1
    cuts: list of lists containing (pi, const) for each stage t
    warmstart: optional dictionary of starting values for x
    
    Returns:
      x_values: [x_0, x_1, x_2, x_3]
      stage_costs: list of stage costs [c_0*x_0, c_1*x_1, ...]
      total_cost: sum of stage costs
    """
    model = pyo.ConcreteModel()
    
    # Indices: 0..T-1
    model.T = range(T)
    
    # Decision variables
    model.x = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    # Optional warmstart
    if warmstart is not None:
        for t in model.T:
            if t in warmstart:
                model.x[t].value = warmstart[t]
    
    # Capacity constraints: x_t <= b_t
    def capacity_rule(m, t):
        return m.x[t] <= scenario[t][1]  # b_t
    model.capacity_con = pyo.Constraint(model.T, rule=capacity_rule)
    
    # Add Benders-type cuts using a Block for each t_idx
    # We must reference x[t_idx] on the *parent model*, i.e. m.model().x[t_idx].
    def benders_cuts_rule(m, t_idx):
        block = pyo.Block()
        block.cons = pyo.ConstraintList()
        # If there are no cuts for t_idx or t_idx==0 or t_idx==T, we might skip
        # But let's still iterate:
        for (pi, const) in cuts[t_idx]:
            # Important: use m.model().x[t_idx] to access the parent model's variable
            block.cons.add(pi * m.model().x[t_idx] <= -const)
        return block
    
    model.benders_block = pyo.Block(model.T, rule=benders_cuts_rule)
    
    # Objective: sum of stage costs c_t*x_t
    def obj_rule(m):
        return sum(scenario[t][0] * m.x[t] for t in m.T)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Solve
    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0  # Mute solver
    solver.solve(model, tee=False)
    
    # Extract solution
    x_values = [pyo.value(model.x[t]) for t in model.T]
    stage_costs = [scenario[t][0] * x_values[t] for t in model.T]
    total_cost = sum(stage_costs)
    
    return x_values, stage_costs, total_cost

def solve_backward(x_values, scenario, cuts):
    """
    After the forward pass gives a solution (x_values), we do a mock backward pass
    from stage T down to stage 1, deriving duals w.r.t x_t. 
    This toy code just 'guesses' pi_t.
    
    x_values: list of x_t from the forward pass
    scenario: list of (c_t, b_t) for t=0..T-1
    cuts: list of Benders cuts per stage
    
    Returns updated cuts.
    """
    # Typically, you'd solve subproblems to get duals. Here, we fake them:
    for t in reversed(range(1, T)):
        pi_t = 0.1 * scenario[t][0]   # e.g. some fraction of cost coeff
        const = -1.0
        cuts[t].append((pi_t, const))
    return cuts

# --------------------
# Main SDDiP Loop
# --------------------
iteration = 0
while iteration < max_iterations and (UB - LB) > 1e-3:
    iteration += 1
    
    # Forward Step
    scenario_set = generate_scenarios(num_scenarios)
    scenario_costs = []
    solutions = []
    
    for scn in scenario_set:
        xvals, stg_costs, tot_cost = solve_forward(scn, cuts)
        scenario_costs.append(tot_cost)
        solutions.append((xvals, scn))
    
    # Update (sample-based) upper bound
    sample_mean = np.mean(scenario_costs)
    UB = min(UB, sample_mean)
    
    # Backward Step
    for (xvals, scn) in solutions:
        cuts = solve_backward(xvals, scn, cuts)
    
    # Lower Bound Update (very naive approach)
    single_scenario = generate_scenarios(1)[0]
    _, _, tot_cost = solve_forward(single_scenario, cuts)
    LB = max(LB, tot_cost)
    
    print(f"Iteration {iteration}: LB={LB:.4f}, UB={UB:.4f}, Gap={UB - LB:.4f}")

print("SDDiP finished.")
print(f"Final Bounds: LB={LB:.4f}, UB={UB:.4f}, GAP={UB - LB:.4f}")
