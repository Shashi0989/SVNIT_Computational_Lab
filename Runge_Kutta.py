# Aim: Perform the RK 2nd, 3rd and 4th order methods to solve the ODE for Newton's law of cooling using Python Programming
import numpy as np
import matplotlib.pyplot as plt

import os
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
clear_screen()

while True:
    print("Newton's Law of Cooling")

    def Input():
        T0 = (float)(input("Enter the initial Temperature of the body: "))  # Initial temperature e.g., T0 = 100
        t0 = (float)(input("Enter the Start time: "))    # Start time e.g., t0 = 0
        t_end = (float)(input("Enter the End time: "))  # End time e.g., t_end = 30
        k = (float)(input("Enter the value of cooling constant: ")) # Cooling constant e.g., k = 0.2
        dt = 0.1    # Time step
        return T0, t0, t_end, dt,k
    
    def exact_solution(t, T0):
        T_env = 27  # Ambient temperature/Room Temperature
        k = 0.2     # Cooling constant
        T = T_env + (T0 - T_env) * np.exp(-k * t)
        return T
    
    def RK_2nd_order(f, T0, t0, t_end, dt):
        n = int((t_end - t0) / dt) + 1
        t = np.linspace(t0, t_end, n)
        T = np.zeros(n)
        T[0] = T0
        for i in range(1, n):
            k1 = f(t[i-1], T[i-1])
            k2 = f(t[i-1] + dt, T[i-1] + dt * k1)
            T[i] = T[i-1] + (dt / 2) * (k1 + 2*k2)
        return t, T
    
    def RK_3rd_order(f, T0, t0, t_end, dt):
        n = int((t_end - t0) / dt) + 1
        t = np.linspace(t0, t_end, n)
        T = np.zeros(n)
        T[0] = T0
        for i in range(1, n):
            k1 = f(t[i-1], T[i-1])
            k2 = f(t[i-1] + dt / 2, T[i-1] + (dt / 2) * k1)
            k3 = f(t[i-1] + dt, T[i-1] + dt * k2)
            T[i] = T[i-1] + (dt / 4) * (k1 + 2 * (k2 + k3))
        return t, T
    
    def RK_4th_order(f, T0, t0, t_end, dt):
        n = int((t_end - t0) / dt) + 1
        t = np.linspace(t0, t_end, n)
        T = np.zeros(n)
        T[0] = T0
        for i in range(1, n):
            k1 = f(t[i-1], T[i-1])
            k2 = f(t[i-1] + dt / 2, T[i-1] + (dt / 2) * k1)
            k3 = f(t[i-1] + dt / 2, T[i-1] + (dt / 2) * k2)
            k4 = f(t[i-1] + dt, T[i-1] + dt * k3)
            T[i] = T[i-1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return t, T
    
    def cooling_ode(t, T, k, T_env=27):
        T_env = 27  # Ambient temperature
        return -k * (T - T_env)
    
    def calculate(T0, t0, t_end, dt, k):
        t_exact = np.linspace(t0, t_end, 100)
        T_exact = exact_solution(t_exact, T0)      
        f = lambda t, T: cooling_ode(t, T, k)  
        t2, T2 = RK_2nd_order(f, T0, t0, t_end, dt)
        t3, T3 = RK_3rd_order(f, T0, t0, t_end, dt)
        t4, T4 = RK_4th_order(f, T0, t0, t_end, dt)
        return  t2, T2, t3, T3, t4, T4, t_exact, T_exact
    
    def Error(t2, T2, t3, T3, t4, T4, T0):
        E2 = np.abs(T2 - exact_solution(t2, T0))
        E3 = np.abs(T3 - exact_solution(t3, T0))
        E4 = np.abs(T4 - exact_solution(t4, T0))
        return E2, E3, E4
    
    def Error_plot(t2, T2, t3, T3, t4, T4, T0):
        E2, E3, E4 = Error(t2, T2, t3, T3, t4, T4, T0)
        fig, ax = plt.subplots(1, 2, figsize=(18, 14), constrained_layout=True)
        ax[0].plot(t2, T2, 'o-' , label='RK 2nd Order', markersize=1.5)
        ax[0].plot(t3, T3, 'o-' , label='RK 3rd Order', markersize=1.5)
        ax[0].plot(t4, T4, 'o-', label='RK 4th Order', markersize=1.5)
        ax[0].set_xlabel('Time (minutes)')
        ax[0].set_ylabel('Temperature (°C)')
        ax[0].legend()
        ax[0].grid()
        
        ax[1].plot(t2, E2, 'o-' , label='RK2 Error', markersize=1.5)
        ax[1].plot(t3, E3, 'o-', label='RK3 Error', markersize=1.5)
        ax[1].plot(t4, E4, 'o-', label='RK4 Error', markersize=1.5)
        ax[1].set_yscale("linear")
        ax[1].set_xlabel('Time (minutes)')
        ax[1].set_ylabel('Absolute Error')
        ax[1].legend()
        ax[1].grid()        
        # Plotting the results
        plt.suptitle("Error Comparison of RK Methods", fontsize=16)
        plt.show()
        
    def Compare_methods(t2, T2, t3, T3, t4, T4):
        plt.figure(figsize=(14, 12))
        plt.plot(t2, T2, label='RK 2nd Order', marker='o', markersize=1.5)
        plt.plot(t3, T3, label='RK 3rd Order', marker='x', markersize=1.5)
        plt.plot(t4, T4, label='RK 4th Order', marker='s', markersize=1.5)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Temperature (°C)')
        plt.axhline(y=27, color='r', linestyle='--', label='Ambient Temperature (27°C)')
        plt.legend()
        plt.grid()
        plt.title("Comparison of RK Methods")
        plt.show()
        return t2, T2, t3, T3, t4, T4
    
    def Compare_all(t2, T2, t3, T3, t4, T4, t_exact, T_exact):
        print("Solving the ODE using RK methods...")
        fig, ax = plt.subplots(1, 2, figsize=(18, 14), constrained_layout=True)
        ax[0].plot(t_exact, T_exact, label='Exact Solution', color='black', linestyle='--')
        ax[0].plot(t2, T2, label='RK 2nd Order', marker='o', markersize=1.5)
        ax[0].plot(t3, T3, label='RK 3rd Order', marker='x' , markersize=1.5)
        ax[0].plot(t4, T4, label='RK 4th Order', marker='s' , markersize=1.5)
        ax[0].set_xlabel('Time (minutes)')
        ax[0].set_ylabel('Temperature (°C)')
        ax[0].axhline(y=27, color='r', linestyle='--', label='Ambient Temperature (27°C)')
        ax[0].legend()
        ax[0].grid()        
        ax[1].plot(t_exact, T_exact, label='Exact Solution', color='black')
        ax[1].set_xlabel('Time (minutes)')
        ax[1].set_ylabel('Temperature (°C)')
        ax[1].axhline(y=27, color='r', linestyle='--', label='Ambient Temperature (27°C)')
        ax[1].legend()
        ax[1].grid()
        plt.suptitle("Newton's Law of Cooling", fontsize=16)
        plt.show()
        return t2, T2, t3, T3, t4, T4
    
    def animate_solution(t2, T2, t3, T3, t4, T4):
        E2,E3,E4 = Error(t2, T2, t3, T3, t4, T4, T0)
        import matplotlib.animation as animation
        N_steps = (int)(t_end/dt)
        fig, ax = plt.subplots(1,2,figsize=(10, 6), constrained_layout=True)
        line2,= ax[0].plot(t2, T2, 'o-', label='RK 2nd Order', markersize=1.5)
        line3, = ax[0].plot(t3, T3, 'o-', label='RK 3rd Order' , markersize=1.5)
        line4, = ax[0].plot(t4, T4, 'o-', label='RK 4th Order' , markersize=1.5)
        ax[0].set_xlim(0, t_end)
        ax[0].set_ylim(0, T0+5)
        ax[0].set_xlabel("Time (minutes)")
        ax[0].set_ylabel("Temperature (°C)")
        ax[0].axhline(y=27, color='r', linestyle='--', label='Ambient Temperature (27°C)')
        
        ax[0].grid()
        ax[0].legend()       
        Error_line2, = ax[1].plot(t2, E2, 'o-' , label='RK2 Error', markersize=1.5)
        Error_line3, =  ax[1].plot(t3, E3, 'o-', label='RK3 Error', markersize=1.5)
        Error_line4, = ax[1].plot(t4, E4, 'o-', label='RK4 Error', markersize=1.5)
        ax[1].set_xlim(0, t_end)
        ax[1].set_ylim(0, max(max(E2),max(E3),max(E4))+1)
        ax[1].set_yscale("linear")
        ax[1].set_xlabel('Time (minutes)')
        ax[1].set_ylabel('Absolute Error')
        ax[1].legend()
        ax[1].grid()
        
        def update(frame):
            line2.set_data(t2[:frame], T2[:frame])
            line3.set_data(t3[:frame], T3[:frame])
            line4.set_data(t4[:frame], T4[:frame])
            Error_line2.set_data(t2[:frame], E2[:frame])
            Error_line3.set_data(t3[:frame], E3[:frame])
            Error_line4.set_data(t4[:frame], E4[:frame])
            return line2, line3, line4, Error_line2, Error_line3, Error_line4
        
        ani = animation.FuncAnimation(fig, update, frames=N_steps, interval=t_end/2,repeat = False, blit=True)
        plt.suptitle("Newton's Law of Cooling Animation", fontsize=16)
        plt.show()

    n = (int)(input("Enter '1' for next input and '0' to break the operation: "))
    if n==1:
        
        clear_screen()
        print("Choose method:")
        print("1. RK 2nd Order")
        print("2. RK 3rd Order")
        print("3. RK 4th Order")
        print("4. Compare all Methods")
        print("5. Compare all Methods with Exact solution Plot")
        print("6. Error Plot")
        print("7. Animate Solution")
        
        choice = int(input("Enter choice: "))
        T0, t0, t_end, dt, k = Input()
        t2, T2, t3, T3, t4, T4, t_exact, T_exact = calculate(T0, t0, t_end, dt, k)
        f = lambda t, T: cooling_ode(t, T, k)
        if choice == 1:
            t2, T2 = RK_2nd_order(f, T0, t0, t_end, dt)
            plt.plot(t2, T2, label='RK 2nd Order', marker='o')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Temperature (°C)')
            plt.axhline(y=27, color='r', linestyle='--', label='Ambient Temperature (27°C)')
            plt.legend()
            plt.grid()
            plt.title("RK 2nd Order Method")
            plt.show()
            
        elif choice == 2:
            t3, T3 = RK_3rd_order(f, T0, t0, t_end, dt)
            plt.plot(t3, T3, label='RK 3rd Order', marker='x')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Temperature (°C)')
            plt.axhline(y=27, color='r', linestyle='--', label='Ambient Temperature (27°C)')
            plt.legend()
            plt.grid()
            plt.title("RK 3rd Order Method")
            plt.show()
            
        elif choice == 3:
            t4, T4 = RK_4th_order(f, T0, t0, t_end, dt)
            plt.plot(t4, T4, label='RK 4th Order', marker='s')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Temperature (°C)')
            plt.axhline(y=27, color='r', linestyle='--', label='Ambient Temperature (27°C)')
            plt.legend()
            plt.grid()
            plt.title("RK 4th Order Method")
            plt.show()
            
        elif choice == 4:
            Compare_methods(t2, T2, t3, T3, t4, T4)
            
        elif choice == 5:
            Compare_all(t2, T2, t3, T3, t4, T4, t_exact, T_exact)
            
        elif choice == 6:
            Error_plot(t2, T2, t3, T3, t4, T4, T0)
            
        elif choice == 7:
            animate_solution(t2, T2, t3 , T3, t4, T4)
            
        else:
            print("Invalid choice. Please try again.")
            
    elif n==0:
        print("Terminated")
        break
    
    else:
        print("Wrong Choice")

    '''def Input():
        T0 = 100  # Initial temperature
        t0 = 0    # Start time  
        t_end = 30  # End time
        dt = 0.1    # Time step
        k = 0.2   # Cooling constant
        return T0, t0, t_end, dt, k'''