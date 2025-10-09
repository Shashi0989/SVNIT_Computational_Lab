import numpy as np
import matplotlib.pyplot as plt
import math
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
clear_screen()

def simulate_and_analyze_walk(n_steps, step_length, n_simulations):
    """
    Simulates 2D random walks with variable step length, plots one example, 
    and calculates distances.

    Args:
        n_steps (int): The number of steps for each random walk.
        step_length (float): The length of each individual step.
        n_simulations (int): The number of walks to simulate for averaging.
    """
    # --- 1. Simulate and Plot a Single Random Walk ---
    print(f"--- Simulating a Single Walk with {n_steps} Steps (Length: {step_length}) ---")
    
    # Start at the origin (0,0)
    x_positions = [0]
    y_positions = [0]
    
    # Perform the walk
    for _ in range(n_steps):
        angle = 2 * math.pi * np.random.random()
        
        # Take a step of length 'step_length' in that direction
        x_step = step_length * math.cos(angle)
        y_step = step_length * math.sin(angle)
        
        x_positions.append(x_positions[-1] + x_step)
        y_positions.append(y_positions[-1] + y_step)

    # --- 2. Draw the Schematic Diagram ---
    plt.figure(figsize=(8, 8))
    plt.plot(x_positions, y_positions, marker='.', label='Path')
    plt.plot(x_positions[0], y_positions[0], 'go', markersize=10, label='Start')
    plt.plot(x_positions[-1], y_positions[-1], 'ro', markersize=10, label='End')
    plt.title(f'Random Walk ({n_steps} Steps, Step Length {step_length})')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

    # --- 3. Calculate Distance for the Single Walk ---
    final_x, final_y = x_positions[-1], y_positions[-1]
    single_walk_distance = math.sqrt(final_x**2 + final_y**2)
    print(f"Distance from start to end for this single walk: {single_walk_distance:.4f}")
    
    print("\n" + "="*50 + "\n")

    # --- 4. Calculate Average Distance over Many Simulations ---
    print(f"--- Simulating {n_simulations} Walks to Find Average Distance ---")
    final_distances = []
    
    for _ in range(n_simulations):
        x, y = 0, 0
        for _ in range(n_steps):
            angle = 2 * math.pi * np.random.random()
            x += step_length * math.cos(angle)
            y += step_length * math.sin(angle)
        final_distances.append(math.sqrt(x**2 + y**2))
        
    average_distance = np.mean(final_distances)
    
    # Updated theoretical formula: l * sqrt(N)
    theoretical_distance = step_length * math.sqrt(n_steps)
    
    print(f"Simulated Average Distance over {n_simulations} walks: {average_distance:.4f}")
    print(f"Theoretical RMS Distance (l * sqrt(N)): {theoretical_distance:.4f}")

# --- User-Defined Values ---

# Let's use fewer steps to make the plot clearer
N = 200

# Increase the step length to increase the spread
L = 5.0 

# Run the simulation with the new step length
simulate_and_analyze_walk(n_steps=N, step_length=L, n_simulations=5000)