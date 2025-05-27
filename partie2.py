import numpy as np
import matplotlib.pyplot as plt


# 2.2
def sinus_echant(f0, fe, N):
    t = np.arange(N) / fe
    signal = np.sin(2 * np.pi * f0 * t)

    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title(f'Signal sinusoïdal : f0 = {f0} Hz, fe = {fe} Hz, N = {N}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    return signal

f = 5
fe = 500
N = 1000
t = np.arange(N) / fe
x = sinus_echant(f, fe, N) 

def autocorrelation_manual(x):
    N = len(x)
    r = np.zeros(2*N - 1)
    lags = np.arange(-N + 1, N)
    for k in range(-N + 1, N):
        somme = 0
        for n in range(N - abs(k)):
            somme += x[n] * x[n + k] if k >= 0 else x[n - k] * x[n]
        r[k + N - 1] = somme
    return lags, r

# Recalcul avec lags
lags_manual, r_manual = autocorrelation_manual(x)

# Lags et corrélation avec NumPy
r_numpy = np.correlate(x, x, mode='full')
lags_numpy = np.arange(-len(x)+1, len(x))

Te = 1 / fe
tau = lags_manual * Te

diff = r_manual - r_numpy

# Tracé avec axe en secondes
plt.figure(figsize=(10, 4))
plt.plot(tau, diff, '--', label="Ecart autocorrélation (manuel - NumPy)")
plt.title("Comparaison des autocorrélations")
plt.xlabel("Décalage (secondes)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def energie_signal(signal):
    return np.sum(signal ** 2)



rapport = energie_signal(diff) / energie_signal(r_numpy)
print(f"Rapport d'énergie (diff vs autocorrélation NumPy) : {rapport:.4e}")
