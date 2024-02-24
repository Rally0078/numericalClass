#RK4 for 2D systems

def RK42D(f, x0, y0, vx0, vy0, t0 = 0, h = 0.01, tFin = 10):
    x = np.array([])
    y = np.array([])
    vx = np.array([])
    vy = np.array([])
    t = t0
    while t <= tFin:
        x = np.append(x, x0)
        y = np.append(y, y0)
        vx = np.append(vx, vx0)
        vy = np.append(vy, vy0)
        k1, l1, m1, n1 = f(x0, y0, vx0, vy0)
        k2, l2, m2, n2 = f(x0 + h*k1/2, y0 + h*l1/2, vx0 + h*m1/2, vy0 + h*n1/2)
        k3, l3, m3, n3 = f(x0 + h*k2/2, y0 + h*l2/2, vx0 + h*m2/2, vy0 + h*n2/2)
        k4, l4, m4, n4 = f(x0 + h*k3, y0 + h*l3, vx0 +h*m3, vy0 + h*n3)

        x0 = x0 + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y0 = y0 + h/6 * (l1 + 2 * l2 + 2 * l3 + l4)
        vx0 = vx0 + h/6 * (m1 + 2 * m2 + 2 * m3 + m4)
        vy0 = vy0 + h/6 * (n1 + 2 * n2 + 2 * n3 + n4)
        t = t + h
    return x, y, vx, vy

#RK4 for any dimensional system with multiple ODE functions passed in an array as parameters
def RK4(initValue, initTime, funcs, stepSize = 0.001, stopTime = 10.0):
    k1 = np.zeros(shape=(len(funcs)))
    k2 = np.zeros(shape=(len(funcs)))
    k3 = np.zeros(shape=(len(funcs)))
    k4 = np.zeros(shape=(len(funcs)))
    y = np.zeros(shape=(len(funcs), 1))
    initValue = np.array(initValue, dtype=np.float64)
    y[:, 0] = np.array(initValue)
    print(y.shape)
    step = 0
    time = initTime
    timeArray = np.array([])
    
    y_step = np.array(initValue)
    #y_step = np.reshape()
    print(y_step.shape)

    while time < stopTime:
        for idx, func in enumerate(funcs):
            k1[idx] = func(y[idx, step], time)
            k2[idx] = func(y[idx, step] + stepSize * k1[idx]/2, time + stepSize/2)
            k3[idx] = func(y[idx, step] + stepSize * k2[idx]/2, time + stepSize/2)
            k4[idx] = func(y[idx, step] + stepSize * k3[idx], time + stepSize)
            step = step + 1
            time = time + stepSize
            timeArray = np.append(timeArray, time)
            y_step[idx] = y_step[idx] + stepSize/6 * (k1[idx] + 2 * k2[idx] + 2 * k3[idx] + k4[idx])
            y = np.column_stack((y, y_step))
    return y, timeArray