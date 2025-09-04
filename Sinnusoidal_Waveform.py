import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
x = np.linspace(-2*np.pi, 2*np.pi, 400)

def Plot_Sinusoidal():
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.figure()
    plt.plot(x, y1, label = 'sin x')
    plt.plot(x, y2, label = 'cos x')
    plt.xlabel("Position (x)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.show()
    
def Animate_Sinusoidal():
    y1=np.zeros(400)
    y2=np.zeros(400)
    global x
    i=0
    fig, ax = plt.subplots()
    line_sin, = ax.plot([],[], lw=2, color='blue', label = 'sin x')
    line_cos, = ax.plot([],[], lw=2, color='orange', label = 'cos x')
    point_sin, = ax.plot([], [], 'ro', markersize=6,color='blue')  # point for animation
    point_cos, = ax.plot([], [], 'ro', markersize=6,color='orange')  # point for animation
    ax.set_xlim(-2*np.pi, 2*np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Sinusoidal Waveform Animation")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Amplitude")
    ax.grid()
    plt.legend()
    print("Animating...")
    def Animate(frame):
        nonlocal i
        if i<400:
            y1[i] = np.sin(x[i])
            y2[i] = np.cos(x[i])
            line_sin.set_data(x[:i],y1[:i])
            line_cos.set_data(x[:i],y2[:i])    
            point_sin.set_data([x[i]], [y1[i]])
            point_cos.set_data([x[i]], [y2[i]])
            i += 1
        return line_sin,line_cos,point_sin, point_cos
    ani = animation.FuncAnimation(fig, Animate, frames=400, blit=True, interval=50, repeat=True)
    plt.show()
    
print("Choose an option: ")
print("1. Plot of Sinusoidal Waveform")
print("2. Animation of Sinusoidal waveform")
n = (int)(input("Enter your option '1' or '2': "))
if n==1:
    Plot_Sinusoidal()
elif n==2:
    Animate_Sinusoidal()
else:
    print("Wrong Choice")