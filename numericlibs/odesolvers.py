import numpy as np
def RK4Full(f, x0, t0 = 0.0, h = 0.01, tFin = 10.0):
    y = np.zeros(shape=(len(x0), 1))
    k1 = k2 = k3 = k4 = np.zeros(shape=(len(x0)))
    y[:, 0] = x0
    t = t0
    timeArray = np.array([])
    while t < tFin:
        timeArray = np.append(timeArray, t)
        k1 = f(x0, t)
        k2 = f(x0 + h*k1/2, t + h/2)
        k3 = f(x0 + h*k2/2, t + h/2)
        k4 = f(x0 + h*k3, t + h)
        x0 = x0 + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        y = np.column_stack((y, x0))
    timeArray = np.append(timeArray, t)
    return y, timeArray