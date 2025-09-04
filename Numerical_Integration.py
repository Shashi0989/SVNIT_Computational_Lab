#Aim : The length L of the suspending cable of a suspension bridge is given by the formula L = 2 * integral from 0 to a of sqrt( 1 + (4h^2 x^2)/(a^4)) dx, where h is the height of the bridge towers and a is half the distance between the towers. Write a Python program to compute L using the trapezoidal rule, simpson's 1/3 rule and simpson's 3/8 rule. Compare the results.
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
clear_screen()

while True:
    clear_screen()
    def integrand(x, h, a):
        return np.sqrt(1 + (4 * h**2 * x**2) / (a**4))

    def exact_length(h, a):
        L, _ = spi.quad(integrand, 0, a, args=(h, a))
        return 2 * L
    
    # ========================
    # (1) Trapezoidal Rule
    # ========================
    
    def trapezoidal_rule(h, a, n):
        x = np.linspace(0, a, n+1)
        dx = a / n
        y = integrand(x, h, a)
        integral = dx * (y[0] + 2 * np.sum(y[1:n]) + y[n]) / 2
        L = 2 * integral
        return L
    
    # ========================
    # (2) Simpson's 1/3 Rule
    # ========================
        
    def simpsons_one_third_rule(h, a, n):
        if n % 2 == 1:
            n += 1  # n must be even for Simpson's 1/3 rule
        x = np.linspace(0, a, n+1)
        dx = a / n
        y = integrand(x, h, a)
        integral = dx / 3 * (y[0] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]) + y[n])
        L = 2 * integral
        return L
    
    # ========================
    # (3) Simpson's 3/8 Rule
    # ========================
    
    def simpsons_three_eighth_rule(h, a, n):
        if n % 3 != 0:
            n += 3 - (n % 3)  # n must be a multiple of 3 for Simpson's 3/8 rule
        x = np.linspace(0, a, n+1)
        dx = a / n
        y = integrand(x, h, a)
        sum3 = np.sum(y[1:n][(np.arange(1, n) % 3 != 0)])
        sum2 = np.sum(y[3:n:3])
        integral = 3 * dx / 8 * (y[0] + 3 * sum3 + 2 * sum2 + y[n])
        L = 2 * integral
        return L

    def Error(f_exact, f_approx):
        return abs(f_exact - f_approx)
    
    # =======================
    # Error Comparison Plot
    # =======================
    
    def Error_Plot(h, a, n_values, exact_L):
        trapezoidal_errors = []
        simpsons_one_third_errors = []
        simpsons_three_eighth_errors = []
        
        for n in n_values:
            L_trap = trapezoidal_rule(h, a, n)
            L_simp13 = simpsons_one_third_rule(h, a, n)
            L_simp38 = simpsons_three_eighth_rule(h, a, n)
            trapezoidal_errors.append(Error(exact_L, L_trap))
            simpsons_one_third_errors.append(Error(exact_L, L_simp13))
            simpsons_three_eighth_errors.append(Error(exact_L, L_simp38))
        plt.figure()
        plt.loglog(n_values, trapezoidal_errors, label="Trapezoidal Rule", marker='o')
        plt.loglog(n_values, simpsons_one_third_errors, label="Simpson's 1/3 Rule", marker='s')
        plt.loglog(n_values, simpsons_three_eighth_errors, label="Simpson's 3/8 Rule", marker='^')
        plt.xlabel("Number of Subintervals (n)")
        plt.ylabel("Absolute Error")
        plt.title("Error Analysis of Numerical Integration Methods")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()                

    def Plot_All_Rules(h, a, n):
        # Base function values
        x = np.linspace(0, a, n+1)
        y = integrand(x, h, a)

        fig, ax = plt.subplots(2, 2, figsize=(12,7), constrained_layout=True)
        ax[0][0].plot(x, y, 'k-', label='f(x)')
        ax[0][0].grid()
        ax[0][0].legend(["f(x)"])
        ax[0][0].set_xlabel("x")
        ax[0][0].set_ylabel("f(x)")
        ax[0][0].text(0.05*a, 0.95*max(y), f"Exact Length ≈ {exact_length(h,a):.5f}", color="blue")
        
        # Trapezoidal
        trap_val = trapezoidal_rule(h, a, n) 
        #ax[0][1].fill_between(x, 0, y)
        for i in range(n):
            ax[0][1].plot([x[i], x[i+1]], [y[i], y[i+1]], 'b-', alpha=0.5)
        ax[0][1].grid()
        ax[0][1].legend(["Trapezoidal"])
        ax[0][1].set_xlabel("x")
        ax[0][1].set_ylabel("f(x)")
        ax[0][1].text(0.05*a, 0.95*max(y), f"Length ≈ {trap_val:.5f}", color="blue")
        
        # Simpson's 1/3 Rule
        n13 = n if n % 2 == 0 else n+1
        x13 = np.linspace(0, a, n13+1)
        y13 = integrand(x13, h, a)
        simpson13_val = simpsons_one_third_rule(h, a, n)
        for i in range(0, n13, 2):
            xi = x13[i:i+3]
            yi = y13[i:i+3]
            if len(xi) == 3:
                coeffs = np.polyfit(xi, yi, 2)
                poly = np.poly1d(coeffs)
                x_fine = np.linspace(xi[0], xi[2], 50)
                ax[1][0].plot(x_fine, poly(x_fine), 'g-', alpha=0.7)
        ax[1][0].grid()
        ax[1][0].legend(["Simpson's 1/3"])
        ax[1][0].set_xlabel("x")
        ax[1][0].set_ylabel("f(x)")
        ax[1][0].text(0.05*a, 0.95*max(y13), f"Length ≈ {simpson13_val:.5f}", color="green")
        
        # Simpson's 3/8 Rule
        n38 = n if n % 3 == 0 else n + (3 - n % 3)
        x38 = np.linspace(0, a, n38+1)
        y38 = integrand(x38, h, a)
        simpson38_val = simpsons_three_eighth_rule(h, a, n)
        for i in range(0, n38, 3):
            xi = x38[i:i+4]
            yi = y38[i:i+4]
            if len(xi) == 4:
                coeffs = np.polyfit(xi, yi, 3)
                poly = np.poly1d(coeffs)
                x_fine = np.linspace(xi[0], xi[3], 50)
                ax[1][1].plot(x_fine, poly(x_fine), 'r-', alpha=0.7)
        ax[1][1].grid()
        ax[1][1].legend(["Simpson's 3/8"])
        ax[1][1].set_xlabel("x")
        ax[1][1].set_ylabel("f(x)")
        ax[1][1].text(0.05*a, 0.95*max(y38), f"Length ≈ {simpson38_val:.5f}", color="red")

        # ========================
        # Main Title
        # ========================
        plt.suptitle("Comparison of Numerical Integration Approximations")
        plt.show()

    def Input():
        h=18.0  # height of the bridge towers in meters
        a=80.0  # half the distance between the towers in meters
        n=10    # number of subintervals
        """
        h = float(input("Enter the height of the bridge towers (h) in meters: "))
        a = float(input("Enter half the distance between the towers (a) in meters: "))
        n = int(input("Enter the number of subintervals (n): "))
        """
        return h, a, n
        

    def main():
        h, a, n = Input()
        exact_L = exact_length(h, a)
        L_trap = trapezoidal_rule(h, a, n)
        L_simp13 = simpsons_one_third_rule(h, a, n)
        L_simp38 = simpsons_three_eighth_rule(h, a, n)
        
        print(f"Exact Length of Cable: {exact_L:.6f} meters")
        print(f"Trapezoidal Rule Length: {L_trap:.6f} meters, Error: {Error(exact_L, L_trap):.6f}")
        print(f"Simpson's 1/3 Rule Length: {L_simp13:.6f} meters, Error: {Error(exact_L, L_simp13):.6f}")
        print(f"Simpson's 3/8 Rule Length: {L_simp38:.6f} meters, Error: {Error(exact_L, L_simp38):.6f}")
        
        n_values = [2**i for i in range(1, 11)]
        print("Choose Plot: ")
        print("1. Plot all methods")
        print("2. Error Plot")
        choice =  int(input("Enter choice: "))
        if choice == 1:
            Plot_All_Rules(h, a, n)
        elif choice == 2:
            Error_Plot(h, a, n_values, exact_L)
        else:
            print("Invalid choice. Please try again.")
    m = (int)(input("Enter '1' for input and '0' to break the operation: "))
    if m==1:
        main()
    elif m==0:
        print("Terminated")
        break
    else:
        print("Wrong Choice")
        