import numpy as np
import coefficients78 as coeff

# Load coefficients as global variables
c = coeff.c  # RK78 nodes
a = coeff.a  # RK78 coupling coefficients
b = coeff.b  # 8th-order weights
bh = coeff.bh  # 7th-order weights

def rk78_step(ode_func, t, y, h, rtol, atol):
    # Initialize the stages array
    k = np.zeros((len(y), 13), dtype=np.float64)

    # Compute the first stage
    k[:, 0] = h * np.array(ode_func(t, y), dtype=np.float64)

    # Intermediate stages
    for i in range(1, 13):
        # Compute temporary state y_temp
        y_temp = y.copy()  # Explicitly copy to avoid unintended modifications
        for j in range(i):
            y_temp += a[i + 1].get(j + 1, 0) * k[:, j]  # Accumulate contributions
        # Compute the next stage
        k[:, i] = h * np.array(ode_func(t + c[i + 1] * h, y_temp), dtype=np.float64)

    # Compute 8th-order and 7th-order approximations
    y8 = y + np.dot(k, np.array(list(b.values())))
    y7 = y + np.dot(k, np.array(list(bh.values())))

    # Compute error and scale factor
    scale_factor = max(np.linalg.norm(y7), np.linalg.norm(y8), 1e-12)
    scaled_error = np.linalg.norm((y8 - y7) / scale_factor, ord=np.inf)

    # Adaptive step size update
    h_new = h * (0.8 / scaled_error**0.2 if scaled_error > 0 else 1.5)
    h_new = min(h_new, 1.5 * h)

    return t + h, y8, h_new


def ode78(ode_func, t_span, y0, rtol=1e-12, atol=1e-14, h=None, h_max=None):
    # Initialize time and state arrays
    t = np.float64(t_span[0])
    y = np.array(y0, dtype=np.float64)

    tout = [t]
    yout = [y.copy()]  # Use `.copy()` to ensure the initial state is not modified

    # Determine initial step size
    if h is None:
        h = (t_span[-1] - t_span[0]) / 100
    if h_max is not None:
        h = min(h, h_max)

    # Time-stepping loop
    while t < t_span[-1]:
        if t + h > t_span[-1]:
            h = t_span[-1] - t  # Adjust the final step size

        # Perform one RK78 step
        t, y, h = rk78_step(ode_func, t, y, h, rtol, atol)
        print(t)
        if h_max is not None:
            h = min(h, h_max)  # Ensure step size doesn't exceed h_max

        tout.append(t)
        yout.append(y.copy())  # Append a copy of the state

    # Combine results into a single array
    return np.column_stack((tout, np.array(yout)))



