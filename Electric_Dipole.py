# Aim: To simulate and animate the electric field and equipotential lines of an electric dipole in 2D and 3D space due to charges at user-defined coordinates.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Tuple

# --- Constants ---
K_COULOMB = 8.98755e9  # Coulomb's constant in N·m²/C²

class TwoChargeSimulator:
    """
    A class to calculate and visualize the electric field and potential of two opposite point charges.
    """
    def __init__(self, charge_magnitude: float, pos_charge_coords: Tuple[float, float, float], neg_charge_coords: Tuple[float, float, float]):
        """
        Initializes the simulation with two charges at specified coordinates.

        Args:
            charge_magnitude (float): Magnitude of the positive charge (+q). The negative charge will be -q.
            pos_charge_coords (Tuple): The (x, y, z) coordinates of the positive charge.
            neg_charge_coords (Tuple): The (x, y, z) coordinates of the negative charge.
        """
        if charge_magnitude <= 0:
            raise ValueError("Charge magnitude must be positive.")
        self.q = charge_magnitude
        self.pos_charge_pos = np.array(pos_charge_coords, dtype=float)
        self.neg_charge_pos = np.array(neg_charge_coords, dtype=float)

    def calculate_field_and_potential(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the electric field (Ex, Ey, Ez) and potential (V) on a 3D grid.
        """
        r_pos = np.stack([x - self.pos_charge_pos[0], y - self.pos_charge_pos[1], z - self.pos_charge_pos[2]], axis=-1)
        r_neg = np.stack([x - self.neg_charge_pos[0], y - self.neg_charge_pos[1], z - self.neg_charge_pos[2]], axis=-1)

        r_mag_pos = np.linalg.norm(r_pos, axis=-1) + 1e-9
        r_mag_neg = np.linalg.norm(r_neg, axis=-1) + 1e-9

        v_total = (K_COULOMB * self.q / r_mag_pos) + (K_COULOMB * -self.q / r_mag_neg)
        e_total = (K_COULOMB * self.q * r_pos / r_mag_pos[..., np.newaxis]**3) + (K_COULOMB * -self.q * r_neg / r_mag_neg[..., np.newaxis]**3)
        
        return e_total[..., 0], e_total[..., 1], e_total[..., 2], v_total

    def plot_2d(self, grid_size: float = 1.0, grid_points: int = 50):
        """
        Generates a static 2D plot of the electric field and equipotential lines.
        """
        x_vals = np.linspace(-grid_size, grid_size, grid_points)
        y_vals = np.linspace(-grid_size, grid_size, grid_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)

        Ex, Ey, _, V = self.calculate_field_and_potential(X, Y, Z)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 8))

        v_abs_max = np.max(np.abs(V))
        contour_levels = np.linspace(-v_abs_max * 0.8, v_abs_max * 0.8, 30)
        cont = ax.contour(X, Y, V, levels=contour_levels, cmap='RdBu', linewidths=1.0)
        fig.colorbar(cont, label='Electric Potential (V)')

        ax.streamplot(X, Y, Ex, Ey, color='black', linewidth=0.7, density=1.5, arrowstyle='->', arrowsize=1.2)
        ax.plot(self.pos_charge_pos[0], self.pos_charge_pos[1], 'o', markersize=12, color='red', label='+q')
        ax.plot(self.neg_charge_pos[0], self.neg_charge_pos[1], 'o', markersize=12, color='blue', label='-q')

        ax.set_title(f'Electric Field of Two Charges (q={self.q:.2e} C)', fontsize=16)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        plt.show()

    def animate_2d_movement(self, grid_size: float = 1.0, grid_points: int = 40, frames: int = 120, interval: int = 50):
        """
        Generates a 2D animation showing one charge oscillating. 🧬
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        x_vals = np.linspace(-grid_size, grid_size, grid_points)
        y_vals = np.linspace(-grid_size, grid_points, grid_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        
        # Store initial positions to reset for the animation logic
        initial_pos_charge_pos = self.pos_charge_pos.copy()
        initial_neg_charge_pos = self.neg_charge_pos.copy()

        def update(frame):
            ax.clear()
            # Animate the positive charge oscillating vertically
            oscillation = 0.2 * grid_size * np.sin(2 * np.pi * frame / frames)
            self.pos_charge_pos = initial_pos_charge_pos + np.array([0, oscillation, 0])
            # Keep the negative charge fixed
            self.neg_charge_pos = initial_neg_charge_pos

            Ex, Ey, _, V = self.calculate_field_and_potential(X, Y, Z)
            
            # Plotting
            v_abs_max = K_COULOMB * self.q / (0.1) # Estimate a fixed potential range
            contour_levels = np.linspace(-v_abs_max, v_abs_max, 25)
            ax.contour(X, Y, V, levels=contour_levels, cmap='RdBu', linewidths=1.0)
            ax.streamplot(X, Y, Ex, Ey, color='black', linewidth=0.7, density=1.5)
            ax.plot(self.pos_charge_pos[0], self.pos_charge_pos[1], 'o', markersize=12, color='red', label='+q')
            ax.plot(self.neg_charge_pos[0], self.neg_charge_pos[1], 'o', markersize=12, color='blue', label='-q')
            
            # Formatting
            ax.set_title(f'Field with Oscillating Charge (+q)', fontsize=16)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_xlim(-grid_size, grid_size); ax.set_ylim(-grid_size, grid_size)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='upper right')

        ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
        plt.show()


if __name__ == '__main__':
    # --- User-defined parameters ---
    charge_val = 12e-9  # Charge magnitude in Coulombs (1 nC)
    
    # Define custom coordinates for the charges
    positive_charge_coords = (-0.02, -0.02, 0)
    negative_charge_coords = (0.02, -0.02, 0)

    # --- 1. Generate a STATIC plot with the custom coordinates ---
    print("Generating 2D static plot with charges at custom coordinates...")
    static_sim = TwoChargeSimulator(
        charge_magnitude=charge_val,
        pos_charge_coords=positive_charge_coords,
        neg_charge_coords=negative_charge_coords
    )
    static_sim.plot_2d(grid_size=0.5, grid_points=50)

    # --- 2. Generate an ANIMATED plot with an oscillating charge ---
    print("\nGenerating 2D animation with one charge oscillating...")
    # For the animation, let's start with a standard horizontal dipole
    animation_sim = TwoChargeSimulator(
        charge_magnitude=charge_val,
        pos_charge_coords=(0.15, 0, 0),
        neg_charge_coords=(-0.15, 0, 0)
    )
    animation_sim.animate_2d_movement(grid_size=0.6, frames=100, interval=50)

# Aim: To simulate and visualize the electric field and potential of a dipole consisting of two point charges, with an animated 2D movement of one charge.
