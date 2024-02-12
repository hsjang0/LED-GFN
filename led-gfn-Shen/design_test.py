import design_bench
task = design_bench.make('TFBind8-Exact-v0')

def solve_optimization_problem(x0, y0):
    return x0  # solve a model-based optimization problem

# solve for the best input x_star and evaluate it
x_star = solve_optimization_problem(task.x, task.y)
y_star = task.predict(x_star)