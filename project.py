import numpy as np 
import matplotlib.pyplot as plt
import math

data = np.loadtxt('data.txt', dtype = float, skiprows = 1)

x = data[:, 0]
y = data[:, 1]
f_xy = data[:, 2]

# ============= Function to determine unique y values ===========

def find_uni(y):
    uni_list = []
    for val in y:
        if val not in uni_list:
            uni_list.append(val)
            
    return uni_list

y_uni = find_uni(y)
y_uni = [float(val) for val in y_uni]

# ============= Data visualization =============

def visualize(data, y_uni):
    plt.figure(figsize=(10, 6))

    for y in y_uni:
        data_y = data[data[:, 1] == y]
        plt.plot(data_y[:, 0], data_y[:, 2], label=f'y = {y}', marker='o', linestyle='-', markersize=4)

    plt.title('F(x, y) for different values ​​of y')
    plt.xlabel('x')
    plt.ylabel('F(x, y)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

visualize(data, y_uni)

# =========== Function to calculate the mean ===========    
    
def calc_mean(y_uni, y, f_xy):
    mean = {}

    for uni in y_uni:
        sum = 0
        count = 0
        for i in range(len(y)):
            if y[i] == uni:
                sum += f_xy[i]
                count += 1
        if count > 0:
            mean[uni] = sum / count
        else:
            mean[uni] = 0
    return mean

means = calc_mean(y_uni, y, f_xy)

# =========== Function to calculate the median ===========

def calc_median(y_uni, y, f_xy):
    medians = {}

    for uni in y_uni:
        val_f = []
        for i in range(len(y)):
            if y[i] == uni:
                val_f.append(f_xy[i])
        
        if len(val_f) > 0:
            val_f.sort()
            n = len(val_f)
            if n % 2 == 1:
                med = val_f[n // 2]
            else:
                med = (val_f[n // 2 - 1] + val_f[n // 2]) / 2
            medians[uni] = med
        else:
            medians[uni] = 0
    return medians

median = calc_median(y_uni, y, f_xy)

# ======= Function to calculate the standard deviation ==========

def calc_std(y_uni, y, f_xy, mean):
    stds = {}

    for uni in y_uni:
        f_vals = []
        for i in range(len(y)):
            if y[i] == uni:
                f_vals.append(f_xy[i])

        if len(f_vals) > 1:  # At least 2 values are required to calculate the standard deviation.
            mean_ = mean[uni]
            sum_sq_diff = sum((x - mean_) ** 2 for x in f_vals)
            std = math.sqrt(sum_sq_diff / len(f_vals))
            stds[uni] = std
        else:
            stds[uni] = 0  # If there is only one value, the standard deviation is 0.

    return stds

std_devs = calc_std(y_uni, y, f_xy, means)

# =========== Plotting statistics graph ===========

mean_vals = [means[y] for y in y_uni]
median_values = [median[y] for y in y_uni]
std_vals = [std_devs[y] for y in y_uni]

width = 0.25
x_labels = np.arange(len(y_uni))

plt.figure(figsize=(14, 6)) 
plt.bar(x_labels - width, mean_vals, width, label='Mean', color='red')
plt.bar(x_labels, median_values, width, label='Median', color='green')
plt.bar(x_labels + width, std_vals, width, label='Standard deviation', color='blue')
plt.xlabel('y values')
plt.ylabel('Statistics values')
plt.title('Mean, median and standard deviation for different values of y')
plt.xticks(x_labels, y_uni)
plt.grid()
plt.legend()
plt.show()
# =========== Polynomial interpolation ===========

def lag_interp(x_data, f_data, x):
    n = len(x_data)
    result = 0
    for i in range(n):
        product = 1
        for j in range(n):
            if i != j:
                product*= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += f_data[i] * product
    return result

# Interpolation for the target y
target_y = 0.0
data_y = data[data[:, 1] == target_y]


# Interpolation over x points
x_interp = np.linspace(min(data_y[:, 0]), max(data_y[:, 0]), 500)
f_interp = [lag_interp(data_y[:, 0], data_y[:, 2], xi) for xi in x_interp]

# Visualization of results
plt.figure(figsize=(10, 6))
plt.plot(data_y[:, 0], data_y[:, 2], 'o', label=f'Data points', color='black')
plt.plot(x_interp, f_interp, label=f'Lagrange interpolation', color='red', linestyle='-')
plt.title(f'Lagrange interpolation for y = {target_y}')
plt.xlabel('x')
plt.ylabel('F(x, y)')
plt.legend()
plt.grid(True)
plt.show()

# =========== Spline interpolation ===========

idx = [i for i in range(len(y)) if y[i] == target_y]

x_layer = x[idx]
f_layer = f_xy[idx]

# Sorting
sort_idx = np.argsort(x_layer)
x_layer = x_layer[sort_idx]
f_layer = f_layer[sort_idx]

# Function to determine coefficients Ki
def calc_ki_spline(x, y, y_deriv0=0.0, y_derivn=0.0):
    n = len(x) - 1
    h = x[1] - x[0]

    A = np.zeros((n+1, n+1))
    B = np.zeros(n+1)

    A[0, 0] = 2
    A[0, 1] = 1
    B[0] = 6 * ((y[1] - y[0]) / h - y_deriv0)

    A[n, n-1] = 1
    A[n, n] = 2
    B[n] = 6 * (y_derivn - (y[n] - y[n-1]) / h)

    for i in range(1, n):
        A[i, i-1] = 1
        A[i, i] = 4
        A[i, i+1] = 1
        B[i] = 6 * ((y[i+1] - y[i]) / h - (y[i] - y[i-1]) / h)

    K = np.linalg.solve(A, B)
    return K

# Spline function S(x)
def S(x0, y0, K, h, x):
    n = len(x0) - 1
    for i in range(n):
        if x0[i] <= x <= x0[i+1]:
            break

    xi = x0[i]
    hi = h

    A = (K[i+1] - K[i]) / (6 * hi)
    B = K[i] / 2
    C = (y0[i+1] - y0[i]) / hi - (2 * hi * K[i] + hi * K[i+1]) / 6
    D = y0[i]

    s = A * (x - xi)**3 + B * (x - xi)**2 + C * (x - xi) + D
    return s

# Interpolation
K = calc_ki_spline(x_layer, f_layer)
h = x_layer[1] - x_layer[0]
x_dense = np.linspace(x_layer[0], x_layer[-1], 500)
s_dense = [S(x_layer, f_layer, K, h, xi) for xi in x_dense]

# Data visualization.
plt.figure(figsize=(10, 6))
plt.plot(x_dense, s_dense, label='Cubic spline interpolation', color='green')
plt.plot(x_layer, f_layer, 'o', label='Data points', color='black')
plt.title('Cubic spline interpolation for y = 0.0')
plt.xlabel('x')
plt.ylabel('F(x, y)')
plt.grid(True)
plt.legend()
plt.show()

# =========== Calculation of interpolation error metrics ===========

def calc_i_err(x_points, f_true, f_interp):
    diff = np.abs(f_true - f_interp)
    mae = np.mean(diff)
    mse = np.mean((f_true - f_interp) ** 2)
    max_err = np.max(diff)
    return {'MAE': mae, 'MSE': mse, 'Max': max_err}

lag_err = calc_i_err(x_interp, np.interp(x_interp, data_y[:, 0], data_y[:, 2]), f_interp)
spline_err = calc_i_err(x_interp, np.interp(x_interp, x_layer, f_layer), s_dense)

metrics = ['MAE', 'MSE', 'Max']
lag_vals = [lag_err[m] for m in metrics]
spline_vals = [spline_err[m] for m in metrics]

x = np.arange(len(metrics))
barwidth = 0.35

bars1 = plt.bar(x - barwidth/2, lag_vals, width=barwidth, label='Lagrange', color='red')
bars2 = plt.bar(x + barwidth/2, spline_vals, width=barwidth, label='Spline', color='green')

plt.xticks(x, metrics)
plt.ylabel('Error value')
plt.title('Comparison of interpolation errors')
plt.legend()
plt.grid(True)

# ========== Helper: add values above bars ==========

def add_vals(bars):
    for b in bars:
        hight = b.get_height()
        plt.text(
            b.get_x() + b.get_width()/2, 
            hight + 0.01 * max(hight, 1), 
            f'{hight:.4f}', 
            ha='center', va='bottom', fontsize=9, color='black'
        )

add_vals(bars1)
add_vals(bars2)

plt.tight_layout()
plt.show()

# =========== Comparison of interpolation functions ===========

plt.figure(figsize=(10, 6))
plt.plot(x_layer, f_layer, 'ko', label='Data points')
plt.plot(x_interp, f_interp, '--', label='Lagrange interpolation', color='red')
plt.plot(x_interp, s_dense, '-', label='Spline interpolation', color='blue')
plt.title(f'Comparison of interpolations for y = {target_y}')
plt.xlabel('x')
plt.ylabel('F(x, y)')
plt.grid(True)
plt.legend()
plt.show()

# =========== Linear approximation ===========

x_target = data_y[:, 0]
f_target = data_y[:, 2]

def linear_approx(x, f):
    n = len(x)
    x_sum = sum(x)
    f_sum = sum(f)
    xf_sum = sum([x[i]*f[i] for i in range(n)])
    x_sum2 = sum([x[i]**2 for i in range(n)])

    
    a = (n * xf_sum - x_sum * f_sum) / (n * x_sum2 - x_sum ** 2)
    b = (f_sum - a * x_sum) / n

    def lin_func(x_val):
        return a * x_val + b

    return lin_func, a, b

f_lin, a_lin, b_lin = linear_approx(x_target, f_target)

# =========== Quadratic approximation ===========

def quad_approx(x, f):
    n = len(x)
    x_sum = sum(x)
    x_sum2 = sum([xi ** 2 for xi in x])
    x_sum3 = sum([xi ** 3 for xi in x])
    x_sum4 = sum([xi ** 4 for xi in x])
    f_sum = sum(f)
    xf_sum = sum([x[i] * f[i] for i in range(n)])
    x_sum2f = sum([x[i]**2 * f[i] for i in range(n)])

    A = np.array([
        [n,     x_sum,   x_sum2],
        [x_sum, x_sum2,  x_sum3],
        [x_sum2, x_sum3, x_sum4]
    ])
    B = np.array([f_sum, xf_sum, x_sum2f])

    coeffs = np.linalg.solve(A, B)
    a, b, c = coeffs[2], coeffs[1], coeffs[0]

    def quad_func(x_val):
        return a * x_val**2 + b * x_val + c

    return quad_func, a, b, c

quad_f, quad_a, quad_b, quad_c = quad_approx(x_target, f_target)

# =========== Data visualization for approximations ===========

x_plot = np.linspace(min(x_target), max(x_target), 300)
y_lin_plot = [f_lin(xi) for xi in x_plot]
y_quad_plot = [quad_f(xi) for xi in x_plot]

plt.figure(figsize=(10, 6))
plt.scatter(x_target, f_target, color='black', label='Data points')
plt.plot(x_plot, y_lin_plot, color='red', label='Linear approximation')
plt.plot(x_plot, y_quad_plot, color='blue', label='Quadratic approximation')
plt.title(f'Function approximation for y = {target_y}')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.grid(True)
plt.show()

# Error calculation
def calc_approx_err(x, f_true, f_model):
    f_pred = np.array([f_model(xi) for xi in x])
    diff = f_true - f_pred

    rmse = np.sqrt(np.mean(diff ** 2))
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((f_true - np.mean(f_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return rmse, r2

rmse_lin, r2_lin = calc_approx_err(x_target, f_target, f_lin)
rmse_quad, r2_quad = calc_approx_err(x_target, f_target, quad_f)

approx_metrics = ['RMSE', 'R²']
lin_vals = [rmse_lin, r2_lin]
quad_vals = [rmse_quad, r2_quad]

x = np.arange(len(approx_metrics))
barwidth = 0.35

bars1 = plt.bar(x - barwidth/2, lin_vals, width=barwidth, label='Linear', color='blue')
bars2 = plt.bar(x + barwidth/2, quad_vals, width=barwidth, label='Quadratic', color='red')

plt.xticks(x, approx_metrics)
plt.ylabel('Error values')
plt.title('Comparison of approximation quality (error metrics)')
plt.legend()
plt.grid(True)

add_vals(bars1)
add_vals(bars2)

plt.tight_layout()
plt.show()

# =========== Integral calculation ===========

def integrate(x, f):
    n = len(x)
    if n < 2:
        return 0  # No data available for integral calculation.
    sum = 0
    for i in range(1, n):
        h = x[i] - x[i - 1]
        sum += h * (f[i] + f[i - 1]) / 2
    return sum

print()
print()

lag_integral = integrate(x_interp, f_interp)
spline_integral = integrate(x_dense, s_dense)
y_lin_apr = [f_lin(xi) for xi in x_plot]
lin_integral = integrate(x_plot, y_lin_apr)
y_quad_apr = [quad_f(xi) for xi in x_plot]
quad_integral = integrate(x_plot, y_quad_apr)

methods = ['Lagrange interpolation', 'Spline interpolation', 'Linear approximation', 'Quadratic approximation']
int_vals = [lag_integral, spline_integral, lin_integral, quad_integral]

# Visualization of results
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, int_vals, color=['blue', 'green', 'orange', 'red'])
plt.title(f'Comparision of integral values')
plt.ylabel('Integral values')
plt.grid(axis='y')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', ha='center', va='bottom')

plt.show()


# More accurate method (Simpson’s rule) for convergence comparison
def integrate_s(x, f):
    n = len(x)
    if n < 3 or n % 2 == 0:
        print('The Simpson method only works for an odd number of points >= 3.')
        return None

    h = (x[-1] - x[0]) / (n - 1)
    sum = f[0] + f[-1]

    for i in range(1, n - 1):
        if i % 2 == 0:
            sum += 2 * f[i]
        else:
            sum += 4 * f[i]

    return (h / 3) * sum

def compare_integrals(f, a, b, N_lista):
    trap_vals = []
    simpson_vals = []

    for N in N_lista:
        if N % 2 == 0:
            N += 1  # Simpson’s rule requires an odd number of intervals.

        x = np.linspace(a, b, N)
        y = [f(xi) for xi in x]

        trap = integrate(x, y)
        simpson = integrate_s(x, y)

        trap_vals.append(trap)
        simpson_vals.append(simpson)

    return trap_vals, simpson_vals

N_vals = [5, 11, 21, 51, 101, 201, 501]
a = min(x_target)
b = max(x_target)
trap, simpson = compare_integrals(quad_f, a, b, N_vals)

plt.figure(figsize=(10, 6))
plt.plot(N_vals, trap, label='Trapezoid method', marker='o', color='red')
plt.plot(N_vals, simpson, label='Simpson method', marker='s', color='blue')
plt.title('Accuracy Comparison of Integration Methods')
plt.xlabel('Number of points')
plt.ylabel('Integral values')
plt.grid(True)
plt.legend()
plt.show()

# =========== Calculation of partial derivatives ===========

def calc_deriv(x, f):
    n = len(x)
    derivs = np.zeros(n)

    for i in range(n):
        if i == 0:
            # Forward difference at the beginning
            derivs[i] = (f[i + 1] - f[i]) / (x[i + 1] - x[i])
        elif i == n - 1:
            # Backward difference at the end
            derivs[i] = (f[i] - f[i - 1]) / (x[i] - x[i - 1])
        else:
            # Central difference in the middle
            derivs[i] = (f[i + 1] - f[i - 1]) / (x[i + 1] - x[i - 1])
    return derivs

# For the target layer y = 0.0
data_layer = data[data[:, 1] == target_y]

# Sorting by x
sort_idx = np.argsort(data_layer[:, 0])
x_sorted = data_layer[sort_idx, 0]
f_sorted = data_layer[sort_idx, 2]

derivs_fx = calc_deriv(x_sorted, f_sorted)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(x_sorted, f_sorted, 'o-', label='F(x, y)', markersize=5, color='red')
plt.plot(x_sorted, derivs_fx, 'o--', label='∂F/∂x', markersize=5, color='black')
plt.title(f'Partial derivatives ∂F/∂x evaluated at y = {target_y}')
plt.xlabel('x')
plt.ylabel('Derived values')
plt.grid(True)
plt.legend()
plt.show()

# Error analysis
def calc_err(x, f):
    N_list = [5, 10, 20, 50, 100]

    # Considering the densest case as the reference
    ref_x = np.linspace(min(x), max(x), 1000)
    ref_f = np.interp(ref_x, x, f)
    ref_deriv = calc_deriv(ref_x, ref_f)

    max_errs = []
    mean_errs = []

    for N in N_list:
        x_N = np.linspace(min(x), max(x), N)
        f_N = np.interp(x_N, x, f)
        N_deriv = calc_deriv(x_N, f_N)

        interp_deriv = np.interp(ref_x, x_N, N_deriv)
        err = np.abs(interp_deriv - ref_deriv)

        max_errs.append(np.max(err))
        mean_errs.append(np.mean(err))

    x_pos = np.arange(len(N_list))
    barwidth = 0.35

    fig, ax = plt.subplots()
    bars1 = plt.bar(x_pos - barwidth/2, max_errs, width=barwidth, label='Maximum error', color='red')
    bars2 = plt.bar(x_pos + barwidth/2, mean_errs, width=barwidth, label='Mean error', color='blue')
    plt.xticks(x_pos, N_list)
    plt.xlabel('Number of points (N)')
    plt.ylabel('Error values')
    plt.title('Dependence of the number of points used for the derivative on error values')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    add_vals(bars1)
    add_vals(bars2)

    plt.tight_layout()
    plt.show()

calc_err(x_sorted, f_sorted)

# =========== Determination of monotonicity ===========

def determine_mono(x, f, deriv, threshold = 1e-6):
    intervals = []
    for i in range(len(x) - 1):
        if deriv[i] > threshold:
            type = 'increasing'
        elif deriv[i] < threshold:
            type = 'decreasing'
        else:
            type = 'constant'
        intervals.append((x[i], x[i+1], f[i], f[i+1], type))
    return intervals

intervals_mono = determine_mono(x_sorted, f_sorted, derivs_fx)

# Data visualization
plt.figure(figsize=(10, 6))
kolory = {'increasing': 'red', 'decreasing': 'blue', 'constant': 'gray'}

for x1, x2, y1, y2, type in intervals_mono:
    plt.plot([x1, x2], [y1, y2], color=kolory[type], linewidth=2.5)

plt.plot([], [], color='red', label='Increasing function')
plt.plot([], [], color='blue', label='Decreasing functon')
plt.plot([], [], color='gray', label='Constant function')

plt.legend()
plt.title(f'Monotonicity of the function F(x, y) with respect to y = {target_y}')
plt.xlabel('x')
plt.ylabel('F(x, y)')
plt.grid(True)
plt.show()