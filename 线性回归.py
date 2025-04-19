import numpy as np
import matplotlib.pyplot as plt
import math, copy
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
def compute_cost(x, y, w, b):
    n = len(x_train)
    cost_sum = 0
    for i in range(n):
        f_wb = w*x[i]+b
        cost_sum = cost_sum + (y[i]-f_wb) ** 2
    cost_sum = (1/(2*n))*cost_sum
    return cost_sum
def compute_gradient(x, y, w, b):
    n = len(x_train)
    dj_dw = 0
    dj_db = 0
    for i in range(n):
        dj_dw = dj_dw + (x[i]*w+b-y[i])*x[i]
        dj_db = dj_db + (x[i]*w+b-y[i])
    dj_dw = (1/n)*dj_dw
    dj_db = (1/n)*dj_db
    return dj_dw, dj_db
def gradient_descent(x, y, w_in, b_in, alpha, num, cost_function, gradient_function):
    w = copy.deepcopy(w_in)
    b = copy.deepcopy(b_in)
    J = []
    p_history = []
    w = w_in
    b = b_in
    for i in range(num):
        cost = cost_function(x, y, w, b)
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w-alpha*dj_dw
        b = b-alpha*dj_db
        if i < 100000:
            J.append(cost)
            p_history.append([w, b])
        if i % math.ceil(num/10) == 0:
            print(f"Iteration {i:4}: Cost {J[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J, p_history
w_init = 0
b_init = 0
new_alpha = 1.0e-2
num_try = 10000
w_final, b_final, J, p = gradient_descent(x_train, y_train, w_init, b_init, new_alpha, num_try, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J[:100])
ax2.plot(1000 + np.arange(len(J[1000:])), J[1000:])
#1000 + np.arange(len(J[1000:])),
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()
