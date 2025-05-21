import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
A = 1.0
f = 5.0              # 5 Hz
fs = 1000            # Sampling frequency
T = 1.0              # Duration = 1 second
t = np.arange(0, T, 1/fs)
x = A * np.sin(2 * np.pi * f * t)

# Empirical autocorrelation
auto_emp = np.correlate(x, x, mode='full')
lags = np.arange(-len(x)+1, len(x)) / fs

# Normalize (optional)
auto_emp = auto_emp / np.max(auto_emp)

# Theoretical autocorrelation
tau = lags
auto_th = (A**2 / 2) * np.cos(2 * np.pi * f * tau)

# Normalize same scale
auto_th = auto_th / np.max(auto_th)

# Plot both
plt.plot(tau, auto_emp, label="Empirical (NumPy)")
plt.plot(tau, auto_th, '--', label="Theoretical")
plt.xlabel("Lag (s)")
plt.ylabel("Autocorrelation")
plt.title("Comparison of Theoretical and Empirical Autocorrelation")
plt.grid(True)
plt.legend()
plt.show()
