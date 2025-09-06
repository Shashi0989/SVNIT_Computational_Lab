# Aim: To find the root of a real-valued function using the Newton-Raphson method.
import numpy as np
import os
import matplotlib.pyplot as plt
import sympy as sp
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
clear_screen()
x = sp.symbols('x')
while True:
    clear_screen()
    print("Newton-Raphson Method for Finding Roots of a Real-Valued Function")
    print("------------------------------------------------------------------")
    func_input = input("Enter the function f(x) (use 'x' as the variable, e.g., x**2 - 4): ")
    func = sp.sympify(func_input)
    f = sp.lambdify(x, func, 'numpy')
    
    f_prime = sp.diff(func, x)
    f_prime_func = sp.lambdify(x, f_prime, 'numpy')
    
    x0 = float(input("Enter the initial guess (x0): "))
    tol = float(input("Enter the tolerance level (e.g., 1e-5): "))
    max_iter = int(input("Enter the maximum number of iterations: "))
    iterations = []
    x_n = x0
    for n in range(1, max_iter + 1):
        f_xn = f(x_n)
        f_prime_xn = f_prime_func(x_n)
        if np.isclose(f_prime_xn, 0, atol=1e-14):
            print("Derivative is zero (or numerically too small). No solution found.")
            break
        
        x_n1 = x_n - f_xn / f_prime_xn
        error = abs(x_n1 - x_n)
        
        iterations.append([n, x_n, f_xn, f_prime_xn, x_n1, error])
        
        if error < tol:
            print(f"Converged to {x_n1} after {n} iterations.")
            break
        
        x_n = x_n1
    else:
        print("Maximum iterations reached. No solution found.")
    
    headers = ["Iteration", "x_n", "f(x_n)", "f'(x_n)", "x_(n+1)", "Error"]
    print(tabulate(iterations, headers=headers, floatfmt=".6f"))
    # Determine min and max from iteration history
    x_values_from_iterations = [row[1] for row in iterations] + [iterations[-1][4]]  # x_n and last x_(n+1)
    x_min, x_max = min(x_values_from_iterations), max(x_values_from_iterations)

    # Add a small margin so the root isn’t right on the edge
    margin = 0.5 * (x_max - x_min) if x_max != x_min else 1
    x_min -= margin
    x_max += margin
    # Plotting the function and the root
    x_vals = np.linspace(x_min-10, x_max+10, 400)
    y_vals = f(x_vals)
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.scatter([x_n], [f(x_n)], color='red', zorder=5)#, label='Root Approximation = {:.6f}'.format(x_n))
    plt.title('Newton-Raphson Method')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(['f(x)', 'x0 (Initial guess) = {:.6f}'.format(x0), 'Root Approximation = {:.6f}'.format(x_n)])
    plt.grid()
    plt.show()
    cont = input("Do you want to find another root? (y/n): ")
    if cont.lower() != 'y':
        break
    