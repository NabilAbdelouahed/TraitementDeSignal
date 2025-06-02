import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

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


def tracer_spectre_wav(nom_fichier, taille_tranche, indice_debut):
    # Lire le fichier .wav
    fe, signal = wavfile.read(nom_fichier)
    
    # Si signal stéréo, prendre un seul canal
    if signal.ndim > 1:
        signal = signal[:, 0]
    
    # Extraire la tranche
    tranche = signal[indice_debut : indice_debut + taille_tranche]
    
    # Appliquer une fenêtre de Hanning pour réduire les effets de bord
    fenetre = np.hanning(len(tranche))
    tranche_fen = tranche * fenetre

    # Calcul FFT
    N = len(tranche_fen)
    fft_result = np.fft.fft(tranche_fen)
    fft_freqs = np.fft.fftfreq(N, d=1/fe)

    # Densité spectrale d'énergie (module au carré normalisé)
    spectre = np.abs(fft_result[:N//2]) ** 2
    spectre_db = 10 * np.log10(spectre / np.max(spectre) + 1e-12)  # éviter log(0)
    freqs_pos = fft_freqs[:N//2]

    # Affichage
    plt.figure(figsize=(10, 4))
    plt.plot(freqs_pos, spectre_db)
    plt.title("Densité spectrale d’énergie")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

tracer_spectre_wav("chat_resample.wav", taille_tranche=1024, indice_debut=8000)