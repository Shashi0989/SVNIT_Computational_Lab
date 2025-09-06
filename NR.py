# Aim: To find the root of a real-valued function using the Newton-Raphson method.
import numpy as np
import os
import matplotlib.pyplot as plt
import sympy as sp
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def clear_screen():
    """
    Clears the terminal screen based on the operating system.
    This provides a clean interface for the user.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def validate_function(expr, var):
    """
    Validates if the expression is a valid function of the variable.
    
    Parameters:
    expr (sympy expression): The mathematical expression to validate
    var (sympy symbol): The variable used in the expression
    
    Returns:
    tuple: (is_valid, error_message) where is_valid is boolean and error_message is string
    """
    try:
        # Convert symbolic expression to a callable function
        f = sp.lambdify(var, expr, 'numpy')
        # Test the function with a sample value to ensure it works
        f(0)
        return True, ""
    except Exception as e:
        # Return False and the error message if validation fails
        return False, f"Invalid function: {str(e)}"

def plot_function_and_iterations(f, f_prime, iterations, root, x0):
    """
    Creates a dual-panel plot showing:
    1. The function and the iteration process
    2. The derivative of the function
    
    Parameters:
    f (function): The original function f(x)
    f_prime (function): The derivative function f'(x)
    iterations (list): List of iteration data
    root (float): The estimated root
    x0 (float): The initial guess
    """
    # Determine a suitable range for plotting based on the iterations and root
    x_min = min([iter[1] for iter in iterations] + [root, x0]) - 1
    x_max = max([iter[1] for iter in iterations] + [root, x0]) + 1
    
    # Generate x values for plotting
    x_vals = np.linspace(x_min, x_max, 400)
    # Calculate corresponding y values for the function and its derivative
    y_vals = f(x_vals)
    y_prime_vals = f_prime(x_vals)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot the function and iterations on the first subplot
    ax1.plot(x_vals, y_vals, label='f(x)', color='blue')
    ax1.axhline(0, color='black', lw=0.5, ls='--')  # Horizontal line at y=0
    ax1.axvline(0, color='black', lw=0.5, ls='--')  # Vertical line at x=0
    
    # Plot the iterations as points and connecting lines
    for i, iter in enumerate(iterations):
        if i < len(iterations) - 1:
            # Draw a line from the current point to the next x-intercept
            ax1.plot([iter[1], iterations[i+1][1]], [iter[2], 0], 'ro--', lw=1, ms=4, alpha=0.7)
        # Plot the current point
        ax1.plot(iter[1], iter[2], 'ro', alpha=0.7)
    
    # Format the root value for display in the legend
    root_formatted = f"{root:.6f}".rstrip('0').rstrip('.')
    
    # Mark the root with a green dot and include the value in the legend
    ax1.plot(root, f(root), 'go', markersize=8, label=f'Root (x ≈ {root_formatted})')
    ax1.set_title('Newton-Raphson Method: Function and Iterations')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot the derivative on the second subplot
    ax2.plot(x_vals, y_prime_vals, label="f'(x)", color='green')
    ax2.axhline(0, color='black', lw=0.5, ls='--')
    ax2.axvline(0, color='black', lw=0.5, ls='--')
    
    # Add a vertical line at the root position on the derivative plot
    ax2.axvline(root, color='red', linestyle=':', alpha=0.7, label=f'Root position (x ≈ {root_formatted})')
    ax2.set_title("Derivative of the Function")
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()

def plot_convergence(iterations, root):
    """
    Plots the error convergence on a logarithmic scale.
    
    Parameters:
    iterations (list): List of iteration data containing error values
    root (float): The estimated root value
    """
    # Extract error values and iteration numbers
    errors = [iter[5] for iter in iterations]
    iterations_num = [iter[0] for iter in iterations]
    
    # Format the root value for display in the title
    root_formatted = f"{root:.6f}".rstrip('0').rstrip('.')
    
    # Create a new figure for the convergence plot
    plt.figure(figsize=(10, 6))
    # Plot errors on a logarithmic scale
    plt.semilogy(iterations_num, errors, 'bo-', lw=2, ms=6)
    plt.title(f'Convergence of Newton-Raphson Method (Root: x ≈ {root_formatted})')
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    # Add grid lines for better readability
    plt.grid(True, which="both", ls="--")
    plt.show()

def main():
    """
    Main function that orchestrates the entire Newton-Raphson process.
    Handles user input, validation, computation, and visualization.
    """
    while True:
        clear_screen()
        print("=" * 60)
        print("NEWTON-RAPHSON METHOD FOR FINDING ROOTS")
        print("=" * 60)
        
        # Get function input with validation
        while True:
            func_input = input("Enter the function f(x) (use 'x' as the variable, e.g., x**2 - 4): ")
            try:
                # Parse the input string into a symbolic expression
                func = sp.sympify(func_input)
                # Validate the function
                is_valid, error_msg = validate_function(func, x)
                if is_valid:
                    break
                else:
                    print(f"Error: {error_msg}")
                    print("Please try again.")
            except sp.SympifyError:
                print("Invalid mathematical expression. Please try again.")
        
        # Create function and its derivative
        f = sp.lambdify(x, func, 'numpy')  # Convert to callable function
        f_prime = sp.diff(func, x)  # Symbolically differentiate
        f_prime_func = sp.lambdify(x, f_prime, 'numpy')  # Convert derivative to callable function
        
        # Display the function and its derivative to the user
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
        iterations = []  # Store iteration data
        x_n = x0  # Start with initial guess
        converged = False  # Track convergence status
        
        for n in range(1, max_iter + 1):
            try:
                # Evaluate function and derivative at current point
                f_xn = f(x_n)
                f_prime_xn = f_prime_func(x_n)
                
                # Handle near-zero derivatives to prevent division by zero
                if abs(f_prime_xn) < 1e-10:
                    print(f"Warning: Derivative is near zero at x = {x_n:.6f}. Method may fail.")
                    # Add a small perturbation to avoid division by zero
                    f_prime_xn = f_prime_xn + 1e-10 if f_prime_xn >= 0 else f_prime_xn - 1e-10
                
                # Calculate next approximation using Newton-Raphson formula
                x_n1 = x_n - f_xn / f_prime_xn
                # Calculate error (difference between successive approximations)
                error = abs(x_n1 - x_n)
                
                # Store iteration data
                iterations.append([n, x_n, f_xn, f_prime_xn, x_n1, error])
                
                # Check for convergence
                if error < tol:
                    print(f"\nConverged to {x_n1:.8f} after {n} iterations.")
                    converged = True
                    root = x_n1  # Store the final root value
                    break
                
                # Update current approximation for next iteration
                x_n = x_n1
                
            except (ValueError, ZeroDivisionError) as e:
                print(f"Error in iteration {n}: {str(e)}")
                root = x_n  # Store the best approximation as root
                break
        
        # Handle case where maximum iterations reached without convergence
        if not converged:
            root = x_n  # Store the best approximation as root
            print(f"\nMaximum iterations reached. Best approximation: {root:.8f}")
        
        # Display results in a table if iterations were performed
        if iterations:
            headers = ["Iteration", "x_n", "f(x_n)", "f'(x_n)", "x_(n+1)", "Error"]
            print("\n" + tabulate(iterations, headers=headers, floatfmt=".6f"))
            
            # Display the final root value
            root_formatted = f"{root:.8f}".rstrip('0').rstrip('.')
            print(f"\nFinal root approximation: x ≈ {root_formatted}")
            
            # Plot the results if iterations were successful
            try:
                plot_function_and_iterations(f, f_prime_func, iterations, root, x0)
                plot_convergence(iterations, root)
            except Exception as e:
                print(f"Error in plotting: {str(e)}")
        
        # Ask if user wants to continue with another function
        while True:
            cont = input("\nDo you want to find another root? (y/n): ").lower()
            if cont in ['y', 'n', 'yes', 'no']:
                break
            print("Please enter 'y' or 'n'.")
        
        # Exit if user doesn't want to continue
        if cont in ['n', 'no']:
            print("Thank you for using the Newton-Raphson Method!")
            break

if __name__ == "__main__":
    # Define the symbolic variable x for use in symbolic operations
    x = sp.symbols('x')
    # Start the main program
    main()