import numpy as np

def cubic(x, x_pts, y_pts):
    h = x_pts[1] - x_pts[0]
    N = len(x_pts)
    A = np.zeros(shape=(N,N))
    A[0,0] = A[-1,-1] = 1.0
    for i in range(1,N-1):
        A[i,i] = 4.0
        A[i,i-1] = A[i,i+1] = 1.0
    B = np.zeros(shape=(N))
    for i in range(1,N-1):
        B[i] = (y_pts[i+1] - 2.0 * y_pts[i] + y_pts[i-1])
    X = np.linalg.inv(A) @ B
    d = np.zeros(shape=(N))
    b = np.zeros(shape=(N))
    for i in range(N-1):
        d[i] = (X[i+1] - X[i])/(3*h)
        b[i] = (y_pts[i+1] - y_pts[i] -X[i]*h**2 -d[i]*h**3)/h
    for i in range(N-1):
        print(f"x={x}, x_pts={x_pts[i]}")
        if(x >= x_pts[i] and x < x_pts[i+1]):
            return y_pts[i] + b[i] * (x - x_pts[i]) + X[i] * (x - x_pts[i])**2 + d[i] * (x - x_pts[i])**3
        
if __name__ == "__main__":
    x = np.array([-1, 1, 3, 5])
    y = np.sin(x)
    angle = np.pi * 2/6
    y_2 = cubic(angle, x, y)
    print(f"{np.sin(angle)}")
    print(f"{y_2}")
    


