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

def validate_function(expr, var):
    """Validate if the expression is a valid function of the variable"""
    try:
        f = sp.lambdify(var, expr, 'numpy')
        # Test the function with a sample value
        f(0)
        return True, ""
    except Exception as e:
        return False, f"Invalid function: {str(e)}"

def plot_function_and_iterations(f, f_prime, iterations, root, x0):
    """Plot the function, its derivative, and the iteration process"""
    # Determine a suitable range for plotting
    x_min = min([iter[1] for iter in iterations] + [root, x0]) - 1
    x_max = max([iter[1] for iter in iterations] + [root, x0]) + 1
    
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = f(x_vals)
    y_prime_vals = f_prime(x_vals)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot the function and iterations
    ax1.plot(x_vals, y_vals, label='f(x)', color='blue')
    ax1.axhline(0, color='black', lw=0.5, ls='--')
    ax1.axvline(0, color='black', lw=0.5, ls='--')
    
    # Plot the iterations
    for i, iter in enumerate(iterations):
        if i < len(iterations) - 1:
            ax1.plot([iter[1], iterations[i+1][1]], [iter[2], 0], 'ro--', lw=1, ms=4)
        ax1.plot(iter[1], iter[2], 'ro')
    
    ax1.plot(root, f(root), 'go', markersize=8, label='Root')
    ax1.set_title('Newton-Raphson Method: Function and Iterations')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot the derivative
    ax2.plot(x_vals, y_prime_vals, label="f'(x)", color='green')
    ax2.axhline(0, color='black', lw=0.5, ls='--')
    ax2.axvline(0, color='black', lw=0.5, ls='--')
    ax2.set_title("Derivative of the Function")
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_convergence(iterations):
    """Plot the error convergence"""
    errors = [iter[5] for iter in iterations]
    iterations_num = [iter[0] for iter in iterations]
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(iterations_num, errors, 'bo-', lw=2, ms=6)
    plt.title('Convergence of Newton-Raphson Method')
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.grid(True, which="both", ls="--")
    plt.show()

def main():
    while True:
        clear_screen()
        print("=" * 60)
        print("NEWTON-RAPHSON METHOD FOR FINDING ROOTS")
        print("=" * 60)
        
        # Get function input with validation
        while True:
            func_input = input("Enter the function f(x) (use 'x' as the variable, e.g., x**2 - 4): ")
            try:
                func = sp.sympify(func_input)
                is_valid, error_msg = validate_function(func, x)
                if is_valid:
                    break
                else:
                    print(f"Error: {error_msg}")
                    print("Please try again.")
            except sp.SympifyError:
                print("Invalid mathematical expression. Please try again.")
        
        # Create function and its derivative
        f = sp.lambdify(x, func, 'numpy')
        f_prime = sp.diff(func, x)
        f_prime_func = sp.lambdify(x, f_prime, 'numpy')
        
        # Display the function and its derivative
        print(f"\nFunction: f(x) = {func}")
        print(f"Derivative: f'(x) = {f_prime}")
        
        # Get initial guess with validation
        while True:
            try:
                x0 = float(input("Enter the initial guess (x0): "))
                break
            except ValueError:
                print("Invalid number. Please enter a valid numeric value.")
        
        # Get tolerance with validation
        while True:
            try:
                tol = float(input("Enter the tolerance level (e.g., 1e-5): "))
                if tol <= 0:
                    print("Tolerance must be positive. Please try again.")
                else:
                    break
            except ValueError:
                print("Invalid number. Please enter a valid numeric value.")
        
        # Get maximum iterations with validation
        while True:
            try:
                max_iter = int(input("Enter the maximum number of iterations: "))
                if max_iter <= 0:
                    print("Number of iterations must be positive. Please try again.")
                else:
                    break
            except ValueError:
                print("Invalid number. Please enter a valid integer value.")
        
        # Perform Newton-Raphson iterations
        iterations = []
        x_n = x0
        converged = False
        
        for n in range(1, max_iter + 1):
            try:
                f_xn = f(x_n)
                f_prime_xn = f_prime_func(x_n)
                
                if abs(f_prime_xn) < 1e-10:
                    print(f"Warning: Derivative is near zero at x = {x_n:.6f}. Method may fail.")
                    # Try to add a small perturbation
                    f_prime_xn = f_prime_xn + 1e-10 if f_prime_xn >= 0 else f_prime_xn - 1e-10
                
                x_n1 = x_n - f_xn / f_prime_xn
                error = abs(x_n1 - x_n)
                
                iterations.append([n, x_n, f_xn, f_prime_xn, x_n1, error])
                
                if error < tol:
                    print(f"\nConverged to {x_n1:.8f} after {n} iterations.")
                    converged = True
                    break
                
                x_n = x_n1
                
            except (ValueError, ZeroDivisionError) as e:
                print(f"Error in iteration {n}: {str(e)}")
                break
        
        if not converged:
            print(f"\nMaximum iterations reached. Best approximation: {x_n:.8f}")
        
        # Display results in a table
        if iterations:
            headers = ["Iteration", "x_n", "f(x_n)", "f'(x_n)", "x_(n+1)", "Error"]
            print("\n" + tabulate(iterations, headers=headers, floatfmt=".6f"))
            
            # Plot the results
            try:
                plot_function_and_iterations(f, f_prime_func, iterations, x_n, x0)
                plot_convergence(iterations)
            except Exception as e:
                print(f"Error in plotting: {str(e)}")
        
        # Ask if user wants to continue
        while True:
            cont = input("\nDo you want to find another root? (y/n): ").lower()
            if cont in ['y', 'n', 'yes', 'no']:
                break
            print("Please enter 'y' or 'n'.")
        
        if cont in ['n', 'no']:
            print("Thank you for using the Newton-Raphson Method!")
            break

if __name__ == "__main__":
    # Define the symbolic variable
    x = sp.symbols('x')
    main()