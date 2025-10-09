# Aim: To calculate the value of Pi using the Monte Carlo method and animate the process (circle inside square)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import itertools
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
clear_screen()

def simple_pi_calculation(num_points):
    """
    Simple version without animation for quick calculation
    """
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside_circle) / num_points
    
    print(f"Total points: {num_points:,}")
    print(f"Points inside circle: {np.sum(inside_circle):,}")
    print(f"Estimated π: {pi_estimate:.10f}")
    print(f"Actual π: {np.pi:.10f}")
    print(f"Error: {abs(pi_estimate - np.pi) / np.pi * 100:.4f}%")
    
    return pi_estimate

def animate_pi_calculation(max_points, points_per_frame):
    """
    Animate the Monte Carlo π calculation process with live data generation.
    """
    # --- Set up the figure and axes ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title('Live Monte Carlo Pi Estimation')

    # Plot 1: Points and circle
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_aspect('equal')
    ax1.set_title('Monte Carlo Simulation')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    circle = patches.Circle((0, 0), 1, fill=False, color='black', linewidth=2, alpha=0.7)
    square = patches.Rectangle((-1, -1), 2, 2, fill=False, color='black', linewidth=2, alpha=0.7)
    ax1.add_patch(circle)
    ax1.add_patch(square)

    inside_scatter = ax1.scatter([], [], s=2, color='#d62728', label='Inside')
    outside_scatter = ax1.scatter([], [], s=2, color='#1f77b4', label='Outside')
    ax1.legend(loc='upper right')

    # Plot 2: π estimate convergence
    ax2.set_xlim(0, max_points)
    ax2.set_ylim(2.8, 3.5)
    ax2.axhline(y=np.pi, color='green', linestyle='--', label=f'True π ({np.pi:.6f})')
    ax2.set_title('π Estimate Convergence')
    ax2.set_xlabel('Number of Points')
    ax2.set_ylabel('π Estimate')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    pi_line, = ax2.plot([], [], 'k-', linewidth=1.5, label='Estimated π')
    pi_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, verticalalignment='top',
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # --- State variables for live simulation ---
    points_inside = 0
    total_points = 0
    # Use lists to store points for efficient updating
    x_in, y_in, x_out, y_out = [], [], [], []
    # Store history for the convergence plot
    pi_history = []
    count_history = []
    
    def update(frame):
        """Update function: generates new data and updates the plot."""
        nonlocal points_inside, total_points

        # Stop the animation if the max number of points is reached
        if total_points >= max_points:
            anim.event_source.stop()
            return inside_scatter, outside_scatter, pi_line, pi_text

        # 1. Generate a new batch of random points
        new_x = np.random.uniform(-1, 1, points_per_frame)
        new_y = np.random.uniform(-1, 1, points_per_frame)

        # 2. Check which points fall inside the circle
        is_inside = (new_x**2 + new_y**2) <= 1

        # 3. Update counts and append points to their respective lists
        points_inside += np.sum(is_inside)
        total_points += points_per_frame
        
        x_in.extend(new_x[is_inside])
        y_in.extend(new_y[is_inside])
        x_out.extend(new_x[~is_inside])
        y_out.extend(new_y[~is_inside])
        
        # 4. Calculate the current estimate of π and store it
        current_pi = 4 * points_inside / total_points
        pi_history.append(current_pi)
        count_history.append(total_points)

        # 5. Update the plots with the new data
        inside_scatter.set_offsets(np.column_stack([x_in, y_in]))
        outside_scatter.set_offsets(np.column_stack([x_out, y_out]))
        pi_line.set_data(count_history, pi_history)
        
        # Update text and titles
        error = abs(current_pi - np.pi) / np.pi * 100
        pi_text.set_text(f'Points: {total_points:,}\n'
                         f'π ≈ {current_pi:.6f}\n'
                         f'Error: {error:.4f}%')
        ax1.set_title(f'Monte Carlo Simulation - {total_points:,} points')

        return inside_scatter, outside_scatter, pi_line, pi_text

    # Create the animation
    anim = FuncAnimation(fig, update, frames=itertools.count(), 
                        interval=20, blit=False, repeat=False, 
                        cache_frame_data=False)    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    animate_pi_calculation(max_points=200000, points_per_frame=500)
    
    # For a more accurate calculation (without animation)
    print("\n" + "="*50)
    print("Running high-precision calculation...")
    simple_pi_calculation(100000)