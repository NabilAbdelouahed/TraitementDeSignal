import numpy as np
import matplotlib.pyplot as plt

#Q3.1

# Paramètres du signal
A = 1            # Amplitude
f0 = 50          # Fréquence du signal en Hz
duration = 0.1   # Durée du signal en secondes

# Temps pour le signal analogique (très haute fréquence d’échantillonnage pour simuler l'analogique)
t_continuous = np.linspace(0, duration, 10000)
s_continuous = A * np.cos(2 * np.pi * f0 * t_continuous)

# Fréquences d’échantillonnage
fe_nyquist = 2 * f0          # Fréquence de Nyquist
fe_sup = 5 * f0              # Fréquence supérieure
fe_inf = 0.8 * f0            # Fréquence inférieure

# Fonctions d’échantillonnage
def sample_signal(fe):
    t = np.arange(0, duration, 1/fe)
    s = A * np.cos(2 * np.pi * f0 * t)
    return t, s

# Échantillonnages
t_nyq, s_nyq = sample_signal(fe_nyquist)
t_sup, s_sup = sample_signal(fe_sup)
t_inf, s_inf = sample_signal(fe_inf)

# Tracé
plt.figure(figsize=(15, 8))

plt.subplot(3, 1, 1)
plt.plot(t_continuous, s_continuous, 'gray', label='Signal analogique')
plt.stem(t_nyq, s_nyq, linefmt='b-', markerfmt='bo', basefmt=' ', label='Échantillonné à f = 2f0')
plt.title("Échantillonnage à la fréquence de Nyquist (2f0)")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_continuous, s_continuous, 'gray', label='Signal analogique')
plt.stem(t_sup, s_sup, linefmt='g-', markerfmt='go', basefmt=' ', label='Échantillonné à f = 5f0')
plt.title("Échantillonnage à une fréquence supérieure (>2f0)")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_continuous, s_continuous, 'gray', label='Signal analogique')
plt.plot(t_continuous, A * np.cos(2 * np.pi * 0.2 * f0 * t_continuous), 'blue', label='Signal construit')
plt.stem(t_inf, s_inf, linefmt='r-', markerfmt='ro', basefmt=' ', label='Échantillonné à f < 2f0')
plt.title("Échantillonnage à une fréquence inférieure (<2f0)")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()


#Q3.2

