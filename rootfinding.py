#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# ## 1. Bisection

# In[32]:


a = 1.0
b = 2.0

def f(x):
    return x**2 - 2
i = 0
x = 0
while a<=b:
    x = (a+b)/2.0
    if f(x) > 0:
        b = x
    elif f(x) < 0:
        a = x
    if i > 12:
        break
    i = i + 1

print(f"Root = {x}")


# ## 2. Newton-Raphson

# In[36]:


x_guess = 1
for i in range(15):
    x_new = x_guess
    x_guess = x_guess - f(x_guess)/(2*x_guess)
    if np.abs(x_new - x_guess) < 1e-8:
        break
print(f"Root = {x_guess}")


# ## 3. Secant method

# In[35]:


x = [1,3]

for i in range(1, 15):
    denom = f(x[i]) - f(x[i-1])
    if np.abs(denom) < 1e-12:
        break
    x_val = x[i] - f(x[i]) * (x[i] - x[i-1])/denom
    x.append(x_val)

print(f"Root = {x[-1]}")

