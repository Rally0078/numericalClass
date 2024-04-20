import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
sigma = 0.2
D = 0.6

def PDESolver(a, b, N, tFin):
    h = (b-a)/N
    k = sigma * h**2/D
    x_val = np.zeros(N)
    t_val = np.arange(0, tFin, k)
    
    w = np.zeros(shape=(len(x_val), len(t_val)))
    w[0, :] = w[-1, :] = 1.0 #Boundary values
    
    for j in range(1, len(t_val) -1):
        w[1:-1, j+1] = sigma * w[0:-2, j] + (1-2*sigma)*w[1:-1,j] + sigma*w[2:,j]
    return w
N = 50
a = 0.0
b = 1.0
h = (b-a)/N
k = sigma * h**2/D
x = np.linspace(a,b,N)  #X axis position
soln = PDESolver(a,b,N,5)

temps = soln[:,-1]  #Y axis Temperature at final time

fig, ax = plt.subplots(figsize=(12,8))

#No animation
"""ax.plot(x, temps[:, time_index])"""

#animation
"""
line, = ax.plot([], [], lw = 3)
maxFrames = len(soln[0, :])//30
interval = 200
frameRate = 50
framesSkip = 10

ax.set_xlim(0,1)
ax.set_xticks(np.arange(0,1.1,0.1))
ax.set_ylim(0,1)
ax.set_yticks(np.arange(0,1.75,0.25))
ax.set_xlabel("Position X")
ax.set_ylabel("Temperature")
ax.set_title(f"Time t = {0.0} seconds")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    ax.set_title(f"Time t = {frame * framesSkip * k:.3f} seconds")
    y = soln[:,frame * framesSkip]
    line.set_data(x, y)
    return line, 
anim = animation.FuncAnimation(fig, update, init_func=init, frames=maxFrames, interval=interval, blit=True)
anim.save("PDEtemp1.mp4", writer="ffmpeg",fps=frameRate)
plt.close()
"""
