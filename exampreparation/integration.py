import numpy as np

def trapezoid(f, a, b, N=1000):
    x = np.linspace(a,b,N)
    h = x[1]-x[0]
    integral = h/2 * (f(a) + f(b) + 2 * np.sum(f(x[1:-1])))
    return integral

def simpsonone(f, a, b, N=1000):
    x = np.linspace(a,b,N)
    h = x[1]-x[0]
    integral = h/3 * (f(a) + f(b) + 4 * np.sum(f(x[1:-1:2])) + 2 * np.sum(f(x[2:-1:2])))
    return integral

def simpsonthree(f, a, b, N=1000):
    x = np.linspace(a,b,N)
    h = x[1]-x[0]
    integral = 3 * h/8 * (np.sum(f(x[0:N-3:3])) + 3*np.sum(f(x[1:N-3:3])) + 3 * np.sum(f(x[2:N-3:3])) + np.sum(f(x[3:N-3:3])))
    return integral

def f(x):
    return np.cos(x)
if __name__ == "__main__":
    a = 0
    b = np.pi/2
    print(trapezoid(f, a, b))
    print(simpsonone(f, a, b))
    print(simpsonthree(f, a, b))