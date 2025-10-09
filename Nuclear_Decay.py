# Aim: To simulate nuclear decay using both exact analytic solutions and Monte Carlo methods, and animate the results.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
clear_screen()
while True:
    clear_screen()

    print("Aim: To simulate nuclear decay using both exact analytic solutions and Monte Carlo methods, and animate the results.")
    # ---------------- Parameters ----------------
    N0 = int(input("Enter the initial number of parent nuclei (default 1000): ") or 1000)   #1000 Initial number of parent nuclei
    half_life = float(input("Enter the half-life (default 5.0): ") or 5.0)          # Half-life (time units)
    lambd = np.log(2) / half_life
    total_time = float(input("Enter the total time to simulate (default 30.0): ") or 30.0)        # Total time to simulate
    frames = int(input("Enter the number of animation frames (default 200): ") or 200)             # Animation frames
    num_simulations = int(input("Enter the number of Monte Carlo trials per frame (default 500): ") or 500)    # Monte Carlo trials per frame

    # ---------------- Physics helpers ----------------
    def exact_method(N0, lambd, t):
        """Exact analytic solution for parent & daughter nuclei."""
        N_parent = N0 * np.exp(-lambd * t)
        N_daughter = N0 - N_parent
        return N_parent, N_daughter

    def monte_carlo_method(N0, lambd, t, num_simulations):
        """Average remaining parents/daughters at time t using Monte Carlo."""
        decay_times = np.random.exponential(1 / lambd, (num_simulations, N0))
        decayed = np.sum(decay_times <= t, axis=1)
        return N0 - decayed.mean(), decayed.mean()

    # ---------------- Precompute exact curves ----------------
    t_exact = np.linspace(0, total_time, 400)
    parent_exact, daughter_exact = exact_method(N0, lambd, t_exact)

    # ---------------- Set up figure & axes ----------------
    fig, (ax_exact, ax_mc) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Exact subplot
    ax_exact.plot(t_exact, parent_exact, 'b-', label="Parent (Exact)")
    ax_exact.plot(t_exact, daughter_exact, 'orange', label="Daughter (Exact)")
    ax_exact.set_ylabel("Number of Nuclei")
    ax_exact.set_title("Exact Solution")
    ax_exact.grid(True)
    ax_exact.legend()

    # Monte Carlo subplot (starts empty)
    mc_parent_scatter = ax_mc.scatter([], [], color='b', s=20, alpha=0.6, label="Parent (MC)")
    mc_daughter_scatter = ax_mc.scatter([], [], color='orange', s=20, alpha=0.6, label="Daughter (MC)")
    ax_mc.set_xlim(0, total_time)
    ax_mc.set_ylim(0, N0 * 1.05)
    ax_mc.set_xlabel("Time")
    ax_mc.set_ylabel("Average Number of Nuclei")
    ax_mc.set_title("Monte Carlo Simulation")
    ax_mc.grid(True)
    ax_mc.legend()

    # ---------------- Animation data storage ----------------
    times = []
    parents_mc = []
    daughters_mc = []

    # ---------------- Update function ----------------
    def update(frame):
        t = total_time * frame / (frames - 1)
        p_mc, d_mc = monte_carlo_method(N0, lambd, t, num_simulations)

        times.append(t)
        parents_mc.append(p_mc)
        daughters_mc.append(d_mc)

        mc_parent_scatter.set_offsets(np.column_stack((times, parents_mc)))
        mc_daughter_scatter.set_offsets(np.column_stack((times, daughters_mc)))
        return mc_parent_scatter, mc_daughter_scatter

    # ---------------- Run the animation ----------------
    ani = FuncAnimation(fig, update, frames=frames,
                        interval=100, blit=False, repeat=False)

    plt.tight_layout()
    plt.show()
    n = input("Press 'n' to exit or any other key to run again: ")
    if n.lower() == 'n':
        break