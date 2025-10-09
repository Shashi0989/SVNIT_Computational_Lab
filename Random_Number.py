# Aim: To generate random numbers and visualize their distribution using mid-square,lagged Fibonacci generator and fibonacci generator, with an animated histogram.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def mid_square(seed, n):
    """Generate n random numbers using the mid-square method."""
    numbers = []
    num = seed
    for _ in range(n):
        num_str = str(num**2).zfill(8)  # Square and pad with zeros
        mid = num_str[2:6]  # Extract middle 4 digits
        num = int(mid)
        numbers.append(num / 10000)  # Normalize to [0, 1)
    return np.array(numbers)

def lagged_fibonacci(seed1, seed2, n):
    """Generate n random numbers using the lagged Fibonacci method."""
    numbers = [seed1, seed2]
    for _ in range(2, n):
        next_num = (numbers[-1] + numbers[-2]) % 1
        numbers.append(next_num)
    return np.array(numbers)

def fibonacci_generator(n):
    """Generate n Fibonacci numbers."""
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return np.array(fib)
def animate_histogram(data, bins=30, interval=100):
    """Animate histogram of the data."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(data) // bins)
    bars = ax.bar(np.linspace(0, 1, bins), np.zeros(bins), width=1/bins)

    def update(frame):
        counts, _ = np.histogram(data[:frame], bins=bins, range=(0, 1))
        for bar, count in zip(bars, counts):
            bar.set_height(count)
        return bars

    ani = FuncAnimation(fig, update, frames=len(data), interval=interval, blit=True)
    plt.show()
if __name__ == "__main__":
    n = 1000  # Number of random numbers to generate

    # Mid-Square Method
    mid_square_data = mid_square(seed=5731, n=n)
    animate_histogram(mid_square_data, bins=30, interval=50)

    # Lagged Fibonacci Method
    lagged_fib_data = lagged_fibonacci(seed1=0.5, seed2=0.75, n=n)
    animate_histogram(lagged_fib_data, bins=30, interval=50)

    # Fibonacci Generator (normalized)
    fib_data = fibonacci_generator(n)
    fib_data = (fib_data - np.min(fib_data)) / (np.max(fib_data) - np.min(fib_data))  # Normalize to [0, 1]
    animate_histogram(fib_data, bins=30, interval=50)
