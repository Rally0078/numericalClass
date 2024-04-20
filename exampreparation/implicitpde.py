import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

D=0.6

def implicit(a,b,M,N,tFin):
    h = (b-a)/M
    k = (tFin)/N
    sigma = D*k/h**2

    u = np.zeros(shape=(M,N))
    u[0,:] = u[-1,:] = 1.0
    A = np.zeros(shape=(M-2,M-2))
    for i in range(M-2):
        A[i,i] = (1+2*sigma)
    for i in range(1,M-2):
        A[i,i-1] = A[i-1,i] = -1*sigma
    b = np.zeros(shape=(M-2))
    b[0] = b[-1] = 1.0
    for j in range(1,N):
        u[1:-1, j] = np.linalg.inv(A) @ (u[1:-1, j-1] + sigma * b)
    return u

M = 50
N = 37500
a = 0.0
b = 1.0
x = np.linspace(a,b,M)  #X axis position
soln = implicit(a,b,M,N,5)
k = 5/N
temps = soln[:,-1]  #Y axis Temperature at final time
print(soln.shape)

fig, ax = plt.subplots(figsize=(12,8))

line, = ax.plot([], [], lw = 3)
maxFrames = len(soln[0, :])
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
    y = soln[:,frame]
    line.set_data(x, y)
    return line, 
anim = animation.FuncAnimation(fig, update, init_func=init, frames=maxFrames, interval=interval, blit=True)
anim.save("PDEtemp2.mp4", writer="ffmpeg",fps=frameRate)
plt.close()