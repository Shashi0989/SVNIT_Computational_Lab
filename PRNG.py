import numpy as np
import matplotlib.pyplot as plt
import random

# Use a style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

# --- Generator Functions (Copied from your original script) ---

def mid_square_method(seed, n_digits=4, n=10):
    """
    Your original Mid-Square Method.
    """
    results = []
    x = seed
    for _ in range(n):
        sq = x * x
        sq_str = str(sq).zfill(2 * n_digits)
        mid = len(sq_str) // 2
        next_x = int(sq_str[mid - n_digits//2 : mid + n_digits//2])
        results.append(next_x)
        x = next_x
        if x == 0:
            break
    return results

def lagged_fibonacci_generator(seed_list, m, n=10):
    """
    Your original Lagged Fibonacci (standard j=1, k=2).
    """
    results = seed_list.copy()
    for i in range(2, n): # Note: starts at 2, goes up to n-1
        x_next = (results[i-1] + results[i-2]) % m
        results.append(x_next)
    # This will return a list of length n, but the first 2 are seeds
    return results

def general_fibonacci_generator(seed_list, j, k, m, n=10):
    """
    Your original General Fibonacci.
    """
    results = seed_list.copy()
    # Start loop at k (length of seed list)
    # Loop n-k times to get a total of n items
    for i in range(k, n):
        x_next = (results[i-j] + results[i-k]) % m
        results.append(x_next)
    return results

def builtin_random_generator(seed, n=10):
    """
    Your original Built-in Random.
    """
    random.seed(seed)
    # Generates numbers from 0 to 9999
    return [random.randint(0, 9999) for _ in range(n)]

# --- Main Program Logic ---

def run_comparison():
    """
    Runs all generators with your original parameters.
    """
    n_numbers = 10
    
    # This dictionary now matches the calls from your original image
    results = {
        'Mid-Square': mid_square_method(5731, n_digits=4, n=n_numbers),
        'Lagged Fibonacci': lagged_fibonacci_generator([1, 2], m=100, n=n_numbers),
        'General Fibonacci': general_fibonacci_generator([1, 2, 3, 4, 5], j=3, k=5, m=100, n=n_numbers),
        'Built-in Random': builtin_random_generator(42, n=n_numbers),
    }
    return results

def visualize_randomness(results):
    """
    Visualizes the randomness as a 2x2 grid of line plots,
    just like your original image.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Flatten the 2x2 axes array for easy iteration
    ax_list = axes.flatten()
    
    for ax, (name, numbers) in zip(ax_list, results.items()):
        if not numbers: # Handle collapsed generators
            ax.text(0.5, 0.5, 'Generator collapsed',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes, color='red')
            ax.set_title(name)
            continue
            
        # Plot value vs. step (index)
        ax.plot(numbers, 'o-', markersize=5)
        ax.set_title(name)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, color = 'black', linestyle='--', linewidth=0.5, alpha=0.7)
        
    plt.tight_layout()
    plt.show()

def analyze_quality(results):
    """
    Your original Basic Analysis function.
    """
    print("\n" + "="*50)
    print("KEY PHYSICS INSIGHTS:")
    print("="*50)
    
    for name, numbers in results.items():
        if not numbers:
            print(f"{name}: Generator collapsed.")
            continue
            
        mean = np.mean(numbers)
        std = np.std(numbers)
        unique = len(set(numbers))
        
        print(f"{name}: mean={mean:.2f}, std={std:.2f}, unique={unique}/{len(numbers)}")

# --- Main execution ---
if __name__ == "__main__":
    
    results = run_comparison()
    
    # Use the 2x2 line plot visualization
    visualize_randomness(results)
    
    # Run the statistical analysis
    analyze_quality(results)