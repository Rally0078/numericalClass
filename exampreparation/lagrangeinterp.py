import numpy as np

def L(x, x_vals, j):
    prod = 1
    for i in range(len(x_vals)):
        if j != i:
            prod = prod * (x - x_vals[i])/(x_vals[j] - x_vals[i])
    return prod

def lagrange(x, x_vals, y_vals):
    N = len(x_vals)
    result = 0
    for i in range(len(x_vals)):
        result = result + L(x, x_vals, i) * y_vals[i]
    return result

if __name__ == "__main__":
    x = np.array([-1, 1, 3, 5])
    y = np.sin(x)
    angle = np.pi*2/6.0
    y_2 = lagrange(angle, x, y)
    print(f"{np.sin(angle)}")
    print(f"{y_2}")