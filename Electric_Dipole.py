import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider  # Import the Slider widget

# Use a style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

# Define Coulomb's constant
K_COULOMB = 8.99e9  # N·m²/C²

def get_electric_field(charges: list, points_x: np.ndarray, 
                       points_y: np.ndarray, points_z: np.ndarray) -> tuple:
    """
    Calculates the total electric field vector components and magnitude 
    along a 1D line in space.
    
    Args:
        charges (list): A list of (q, (xq, yq, zq)) tuples.
        points_x (np.array): x-coordinates of the line.
        points_y (np.array): y-coordinates of the line.
        points_z (np.array): z-coordinates of the line.
        
    Returns:
        tuple: (Ex_net, Ey_net, Ez_net, E_total_mag)
    """
    
    Ex_net = np.zeros_like(points_x)
    Ey_net = np.zeros_like(points_y)
    Ez_net = np.zeros_like(points_z)
    
    for q, (xq, yq, zq) in charges:
        Rx = points_x - xq
        Ry = points_y - yq
        Rz = points_z - zq
        
        R_mag_sq = Rx**2 + Ry**2 + Rz**2
        R_mag_sq[R_mag_sq == 0] = 1e-12 
        R_mag_cubed = R_mag_sq**1.5
        
        Ex_net += K_COULOMB * q * Rx / R_mag_cubed
        Ey_net += K_COULOMB * q * Ry / R_mag_cubed
        Ez_net += K_COULOMB * q * Rz / R_mag_cubed
        
    E_total_mag = np.sqrt(Ex_net**2 + Ey_net**2 + Ez_net**2)
    
    return Ex_net, Ey_net, Ez_net, E_total_mag

# --- Main script execution ---
if __name__ == "__main__":
    
    # --- 1. Get User Inputs for Fixed Parameters ---
    q = float(input("Enter the magnitude of each charge in microCoulombs (e.g., 12 for 12 µC): ") or 12) * 1e-6  # Convert to Coulombs
    x1 = float(input("Enter the x-coordinate of the positive charge in cm (e.g., 2 for 2 cm): ") or -2) * 1e-2  # Convert to meters
    x2 = float(input("Enter the x-coordinate of the negative charge in cm (e.g., 2 for 2 cm): ") or 2) * 1e-2  # Convert to meters    
    
    # --- This will be the DYNAMIC parameter ---
    y_initial_cm = float(input("Enter initial y-coord for dipole in cm (default -2 cm): ") or -2)
    
    # --- 2. Define the Line to Plot Along ---
    num_points = 500
    x_line = np.linspace(-0.10, 0.10, num_points)  # -10 cm to +10 cm
    y_line = np.zeros(num_points)                   # y = 0
    z_line = np.zeros(num_points)                   # z = 0

    # --- 3. Set up the Figure and Axes for Plotting ---
    # Create a figure and a main axes object
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Adjust the main plot to make room for the slider at the bottom
    plt.subplots_adjust(bottom=0.25)
    
    # --- 4. Define the initial plot ---
    # We need to calculate the E-field once to create the line object
    y_initial_m = y_initial_cm * 1e-2
    initial_charges = [
        (q, (x1, y_initial_m, 0)),   # Positive charge
        (-q, (x2, y_initial_m, 0))  # Negative charge
    ]
    _, _, _, E_initial = get_electric_field(initial_charges, x_line, y_line, z_line)
    
    # Plot the initial data and store the line object
    line, = ax.plot(x_line * 100, E_initial, 'b-', linewidth=2, label='|E| (N/C)')
    
    # Add static vertical lines for charge x-positions
    ax.axvline(x1 * 100, color='red', linestyle='--', alpha=0.7, 
               label=f'+q x-pos at {x1*100:.1f} cm')
    ax.axvline(x2 * 100, color='blue', linestyle='--', alpha=0.7, 
               label=f'-q x-pos at {x2*100:.1f} cm')
    
    # --- 5. Format the main plot ---
    ax.set_xlabel('Position x (cm)', fontsize=12)
    ax.set_ylabel('Electric Field Magnitude |E| (N/C) [Log Scale]', fontsize=12)
    title_obj = ax.set_title(f'Electric Field Along x-axis (Dipole at y={y_initial_cm:.1f} cm)', fontsize=14)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(which='major', linestyle='--', alpha=0.7, color='gray', linewidth=0.8)

    # --- 6. Create the Slider ---
    # Define the axes for the slider [left, bottom, width, height]
    ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    
    # Create the slider
    y_slider = Slider(
        ax=ax_slider,
        label='Dipole y-pos (cm)',
        valmin=-10.0,  # Min y-position (cm)
        valmax=10.0,   # Max y-position (cm)
        valinit=y_initial_cm,  # Initial value
        valstep=0.1    # Step size
    )

    # --- 7. Define the Update Function (called on slider change) ---
    def update(val):
        # Get the new y-position from the slider
        y_pos_cm = y_slider.val
        y_pos_m = y_pos_cm * 1e-2
        
        # Create the new charge list
        charge_list = [
            (q, (x1, y_pos_m, 0)),
            (-q, (x2, y_pos_m, 0))
        ]
        
        # Recalculate the E-field
        _, _, _, E_total_mag = get_electric_field(charge_list, x_line, y_line, z_line)
        
        # Update the y-data of the existing line
        line.set_ydata(E_total_mag)
        
        # Update the title
        title_obj.set_text(f'Electric Field Along x-axis (Dipole at y={y_pos_cm:.1f} cm)')
        
        # Rescale the y-axis to fit the new data
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        
        # Redraw the canvas
        fig.canvas.draw_idle()

    # --- 8. Attach the update function to the slider ---
    y_slider.on_changed(update)

    # --- 9. Show the plot ---
    plt.show()