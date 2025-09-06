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
        
        if f_prime_xn == 0:
            print("Derivative is zero. No solution found.")
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
    
    # Plotting the function and the root
    x_vals = np.linspace(x0 - 10, x0 + 10, 400)
    y_vals = f(x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.scatter([x_n], [f(x_n)], color='red', zorder=5)
    plt.title('Newton-Raphson Method')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.show()
    cont = input("Do you want to find another root? (y/n): ")
    if cont.lower() != 'y':
        break