# Aim: To  calculate the current flowing through each branch of the circuit using Kirchoff's Voltage law using Four current loops
import numpy as np
import scipy.linalg as sci
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
clear_screen()

while True:
    print("Aim: To  calculate the current flowing through each branch of the circuit using Kirchoff's Voltage law using Four current loops")

    # Defining the coefficient matrix
    A = np.array([[44, -10, -16, 0],
                [10, -43, 6, 12],
                [16, 6, -30, 8],
                [0, 12, 8, -34]])
    # Defining the constants matrix
    B = np.array([20, 0, 12, -40])
    def SciPy(A, B):
        # Solving the linear equations using SciPy
        X = sci.solve(A, B)
        
        print("The currents flowing through each branch are:")
        for i, current in enumerate(X, start=1):
            print(f"Branch {i}: {current:.2f} A")
        print("\n")
        return X

    def LU_decomposition(A, B):
        P, L, U = sci.lu(A)
        y = sci.solve(L, np.dot(P, B))
        x = sci.solve(U, y)
        
        print("Using LU Decomposition, the currents flowing through each branch are:")
        X_lu = x
        for i, current in enumerate(X_lu, start=1):
            print(f"Branch {i}: {current:.2f} A")
        print("\n")
        return x

    # Same thing using Cramer's Rule
    def cramer_rule(A, B):
        det_A = np.linalg.det(A)
        if det_A == 0:
            raise ValueError("The system has no unique solution.")
        
        n = A.shape[0]
        X = np.zeros(n)
        
        for i in range(n):
            A_i = A.copy()
            A_i[:, i] = B
            X[i] = np.linalg.det(A_i) / det_A

        # Solving using Cramer's Rule
        X_cramer = cramer_rule(A, B)
        # Displaying the results
        print("Using Cramer's Rule, the currents flowing through each branch are:")
        for i, current in enumerate(X_cramer, start=1):
            print(f"Branch {i}: {current:.2f} A")
        print("\n")
        return X_cramer
    def main():
        print("Calculating currents using different methods (choose one):\n 1. SciPy \n 2. LU Decomposition \n 3. Cramer's Rule")
        choice = input("Enter your choice (1/2/3): ")
        if choice == '1':
            SciPy(A, B)
        elif choice == '2':
            LU_decomposition(A, B)
        elif choice == '3':
            cramer_rule(A, B)
        else:
            print("Invalid choice.")
    if __name__ == "__main__":
        main()
    cont = input("Do you want to perform another calculation? (y/n): ")
    if cont.lower() != 'y':
        break
