import numpy as np
import matplotlib.pyplot as plt


# 2.1 

def autocorrelation_theorique_sinus(A, f, tau):
    """
    Calcule l'autocorrélation théorique d'un sinus A*sin(2πft)
    
    Paramètres :
    - A : amplitude du sinus
    - f : fréquence en Hz
    - tau : vecteur de lags (décalages temporels) en secondes (numpy array)
    
    Retour :
    - R(tau) : autocorrélation théorique
    """
    return (A ** 2) / 2 * np.cos(2 * np.pi * f * tau)

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

t = np.arange(1000) / 100
x = sinus_echant(5, 500, 1000) 

def autocorrelation_manual(x):
    N = len(x)
    r = np.zeros(2*N - 1)
    for k in range(-N + 1, N):
        somme = 0
        for n in range(N - abs(k)):
            somme += x[n] * x[n + k] if k >= 0 else x[n - k] * x[n]
        r[k + N - 1] = somme
    return r


# Calcul manuel
r_manual = autocorrelation_manual(x)

# Calcul avec NumPy
r_numpy = np.correlate(x, x, mode='full')

# Tracé
plt.figure(figsize=(10, 4))
plt.plot(r_manual, label="Autocorrélation manuelle", linestyle='--')
plt.plot(r_numpy, label="Autocorrélation NumPy", alpha=0.6)
plt.title("Comparaison de l'autocorrélation (valeurs seulement)")
plt.xlabel("Indice")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def autocorrelation_with_lags(x):
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
lags_manual, r_manual = autocorrelation_with_lags(x)

# Lags et corrélation avec NumPy
r_numpy = np.correlate(x, x, mode='full')
lags_numpy = np.arange(-len(x)+1, len(x))

r_sinus = autocorrelation_theorique_sinus(1, 5, lags_numpy/500)

r_manual_norm = r_manual / np.max(r_manual)
r_numpy_norm = r_numpy / np.max(r_numpy)
r_sinus_norm = r_sinus / np.max(np.abs(r_sinus)) 


# Tracé avec abscisses correctes
plt.figure(figsize=(10, 4))
plt.plot(lags_numpy, r_sinus, label='Autocorrélation théorique')
plt.plot(lags_manual, r_manual, label="Autocorrélation manuelle", linestyle='--')
plt.plot(lags_numpy, r_numpy, label="Autocorrélation NumPy", alpha=0.6)
plt.title("Comparaison avec les lags (décalages)")
plt.xlabel("Décalage (échantillons)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
Obtenez-vous ce qui est attendu et si non, pourquoi ?

Non, pas au début car l'autocorrelation est centrée en 0 (symetrique a l'axe des ordonnées) et sur le tracé elle ne l'est pas.
il faut donc calculer les lags pour que le tracé soit correct.
"""