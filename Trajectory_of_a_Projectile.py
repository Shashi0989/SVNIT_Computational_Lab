#Aim : To calculate the trajectory of a projectile (ball) by repeatedly applying the law of motion per small time steps using for loop
import numpy as np
import matplotlib.pyplot as plt

import os
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
clear_screen()

while True:
    v0 = (float)(input("Enter the initial velocity of the projectile: ")) #initial velocity in m/s
    g = 9.81 #acceleration due to gravity in m/s^2
    theta = (float)(input("Enter the initial angle in degrees: ")) #iniial angle in degrees
    theta_rad = np.radians(theta) #convert angle to radians
    vx = v0 * np.cos(theta_rad) #horizontal component of velocity
    vy = v0 * np.sin(theta_rad) #vertical component of velocity
    t0 = 0 #initial time in seconds
    dt = 0.01 #time step in seconds
    T = 2 * v0 * np.sin(theta_rad) / g #total time of flight in seconds
    N_steps = int(T / dt) #number of time steps
    H = v0**2*np.sin(theta_rad)**2/(2*g) #maximum height in meters
    R = v0**2 * np.sin(2*theta_rad) / g #range in meters
    param_text = " "
    def PLotTrajectory():
        print("Calculating trajectory...")
        x = np.zeros(N_steps) #initialize x position array
        y = np.zeros(N_steps) #initialize y position array
        for i in range(N_steps):
            t = t0 + i * dt #current time
            x[i] = vx * t #update x position
            y[i] = vy * t - 0.5 * g * t**2 #update y position
        plt.figure()
        plt.plot(x, y, label='Trajectory', color='blue' if theta >= 0 else 'red')
        plt.title("Projectile Motion Trajectory")
        plt.xlabel("Horizontal Distance (m)")
        plt.ylabel("Vertical Distance (m)")
        plt.xlim(0, max(x) + 5)
        plt.ylim(0, max(y) + 5)
        plt.grid()
        plt.legend(loc="best")
        param_text = f"$v_0$ = {v0} m/s\n$θ_0$ = {theta}°\nRange ≈ {max(x):.2f} m\nMax Height ≈ {max(y):.2f} m\nTime of Flight ≈ {T:.2f} s"
        plt.text(0.05*max(x), 0.95*max(y), param_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.2))
        plt.show()
        
    def AnimateTrajectory():    
        t_array = np.linspace(t0, T, N_steps) #time array
        x = np.zeros(N_steps) #initialize x position array
        y = np.zeros(N_steps) #initialize y position array
        i=0
        v_y = vy #initial vertical velocity
        import matplotlib.animation as animation
        fig, ax = plt.subplots(constrained_layout=True)
        point, = ax.plot([], [], 'ro', markersize=6)  # projectile point
        line, = ax.plot([], [], lw=2) # trajectory line
        text = ax.text(0.05, 0.75, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor="white", alpha=0.2))
        ax.set_xlim(0, R + 10) #set x limits
        ax.set_ylim(0, H + 10) #set y limits
        ax.set_title("Projectile Motion Animation")
        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Vertical Distance (m)")
        ax.grid()
        print("Animating trajectory...")
        def Animate(frame):
            nonlocal i,x,y, v_y
            global t, dt, vx, vy, g, param_text
            param_text = " "
            i=frame
            if i < N_steps:
                t = t_array[i] #current time
                x[i] = vx * t #update x position
                y[i] = vy * t - 0.5 * g * t**2 #update y position
                v_y = v_y - g * dt  # update vertical velocity
                line.set_data(x[:i], y[:i])
                # Update the text dynamically
                point.set_data([x[i]], [y[i]])  # update projectile position
                text.set_text(f"t = {t:.2f} s\n"f"x = {x[i]:.2f} m\n"f"y = {y[i]:.2f} m\n"f"$v_x$ = {vx} m/s\n"f"$v_y$ = {v_y}\n"f"θ = {np.degrees(np.arctan(v_y/vx))}°")
            return line, text, point,
        param_text = f"$v_0$ = {v0} m/s\n$\\theta_0$ = {theta}°\nRange ≈ {R:.2f} m\nMax Height ≈ {H:.2f} m\nTime of Flight ≈ {T:.2f} s"
        plt.text(0.85*R, 0.95*H, param_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.2))
        ani = animation.FuncAnimation(fig, Animate, frames=N_steps, blit=True, interval=50, repeat=False)
        plt.legend(['Trajectory'])
        plt.show()
        
    print("Choose an option:")
    print("1. Plot Trajectory")
    print("2. Animate Trajectory")
    choice = input("Enter your choice (1 or 2): ")
    if choice == '1':
        PLotTrajectory()
    elif choice == '2':
        AnimateTrajectory()
    else:
        print("Invalid choice")
    n = (int)(input("Enter '1' for next input and '0' to break the operation: "))
    if n==1:
        clear_screen()
    elif n==0:
        print("Terminated")
        break
    else:
        print("Wrong Choice")
