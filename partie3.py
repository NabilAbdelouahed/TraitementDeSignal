import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal.windows import hann, hamming, blackman
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
    
    # Appliquer une fenêtre de Hanning
    fenetre = np.hanning(len(tranche))
    tranche_fen = tranche * fenetre

    # FFT
    N = len(tranche_fen)
    fft_result = np.fft.fft(tranche_fen)
    fft_freqs = np.fft.fftfreq(N, d=1/fe)

    # Densité spectrale d’énergie
    spectre = np.abs(fft_result[:N//2]) ** 2
    spectre_db = 10 * np.log10(spectre / np.max(spectre) + 1e-12)
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

    return freqs_pos, spectre_db  # <<< retourne les données


def detecter_fondamentale(freqs, pse):
    idx_min = np.argmax(freqs > 50)
    peaks, _ = find_peaks(pse[idx_min:], height=-40)
    if len(peaks) > 0:
        i_peak = peaks[0] + idx_min
        f0 = freqs[i_peak]
        print(f"Fréquence fondamentale détectée : {f0:.1f} Hz")
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, pse)
        plt.plot(freqs[i_peak], pse[i_peak], "ro", label=f"f₀ = {f0:.1f} Hz")
        plt.title("Densité spectrale d’énergie avec f₀")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Aucun pic détecté.")

# Appel principal
freqs, pse = tracer_spectre_wav("aaa.wav", taille_tranche=4096, indice_debut=20)
detecter_fondamentale(freqs, pse)

def zero_padding(signal, nb_zeros):
    """Ajoute nb_zeros zéros à la fin du signal"""
    return np.hstack((signal, np.zeros(nb_zeros)))

def tracer_spectres_compare(nom_fichier, nb_zeros):
    """Compare les densités spectrales avec et sans zero-padding"""
    # Lire le fichier .wav
    fe, signal = wavfile.read(nom_fichier)
    
    # Si signal stéréo, prendre un seul canal
    if signal.ndim > 1:
        signal = signal[:, 0]

    # Sans padding
    N = len(signal)
    fen = np.hanning(N)
    signal_win = signal * fen
    fft_signal = fft(signal_win)
    freqs = fftfreq(N, d=1/fe)[:N//2]
    pse = 10 * np.log10(np.abs(fft_signal[:N//2])**2 / np.max(np.abs(fft_signal[:N//2])**2) + 1e-12)

    # Avec zero-padding
    signal_pad = zero_padding(signal, nb_zeros)
    N_pad = len(signal_pad)
    fen_pad = np.hanning(N_pad)
    signal_pad_win = signal_pad * fen_pad
    fft_pad = fft(signal_pad_win)
    freqs_pad = fftfreq(N_pad, d=1/fe)[:N_pad//2]
    pse_pad = 10 * np.log10(np.abs(fft_pad[:N_pad//2])**2 / np.max(np.abs(fft_pad[:N_pad//2])**2) + 1e-12)

    # Tracé
    plt.figure(figsize=(12, 4))
    plt.plot(freqs, pse, label="Sans zero-padding")
    plt.plot(freqs_pad, pse_pad, label="Avec zero-padding", linestyle='--')
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("Comparaison des densités spectrales d’énergie")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Q3.3


def reduce_cadence(filename, tranche_size, start_index, factor, out_filename):
    """
    Accélère le signal d'un facteur `factor` :
    - On prend 1 échantillon sur `factor` (décimation).
    - On conserve fs identique (donc la durée est divisée par factor).
    """
    # Lecture du wav
    fs, signal = wavfile.read(filename)
    # On extrait la tranche demandée
    tranche = signal[start_index:start_index + tranche_size]
    # On décime : on prend un échantillon sur 'factor'
    reduced = tranche[::factor]
    # On garde la même fréquence d'échantillonnage
    new_fs = fs
    # On écrit le résultat
    wavfile.write(out_filename, new_fs, reduced.astype(np.int16))
    return reduced, new_fs


def increase_cadence(filename, tranche_size, start_index, factor, out_filename):
    """
    Ralentit le signal d'un facteur `factor` :
    - On répète chaque échantillon `factor` fois (np.repeat).
    - On conserve fs identique (donc la durée est multipliée par factor).
    """
    # Lecture du wav
    fs, signal = wavfile.read(filename)
    # On extrait la tranche demandée
    tranche = signal[start_index:start_index + tranche_size]
    # On répète chaque échantillon factor fois
    upsampled = np.repeat(tranche, factor)
    # On garde la même fréquence d'échantillonnage
    new_fs = fs
    # On écrit le résultat
    wavfile.write(out_filename, new_fs, upsampled.astype(np.int16))
    return upsampled, new_fs


def generate_cadence_variations(filename, tranche_size, start_index):
    """
    Génère plusieurs fichiers .wav accélérés (x2, x3, x4) et ralentis (÷2, ÷3, ÷4)
    dans le dossier 'outputs_cadence/'. Pour chaque facteur f :
      - accéléré : fichier "faster_x{f}.wav"
      - ralenti  : fichier "slower_x{f}.wav"
    """
    acceleration_factors = [2, 3, 4, 5, 6]  # 2x, 3x, 4x plus rapides
    slowing_factors      = [2, 3, 4, 5, 6]  # ÷2, ÷3, ÷4 plus lents

    for f in acceleration_factors:
        out_file = f"outputs_cadence/faster_x{f}.wav"
        reduce_cadence(filename, tranche_size, start_index, f, out_file)
        print(f"[✔] Fichier généré : {out_file} (accéléré x{f})")

    for f in slowing_factors:
        out_file = f"outputs_cadence/slower_x{f}.wav"
        increase_cadence(filename, tranche_size, start_index, f, out_file)
        print(f"[✔] Fichier généré : {out_file} (ralenti ÷{f})")


generate_cadence_variations(filename= "audio-sig.wav", tranche_size= 25600, start_index = 500)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Dictionnaire des fichiers à comparer
file_paths = {
    "Original":            "audio-sig.wav",
    "Faster x2":           "outputs_cadence/faster_x2.wav",
    "Faster x4":           "outputs_cadence/faster_x4.wav",
    "Slower x2":           "outputs_cadence/slower_x2.wav",
    "Slower x4":           "outputs_cadence/slower_x4.wav"
}

# Création de la figure avec 5 sous-plots (une rangée de 5 graphiques)
fig, axs = plt.subplots(5, 1, figsize=(10, 15))

for ax, (label, path) in zip(axs, file_paths.items()):
    if not os.path.exists(path):
        # Si le fichier n'existe pas, on l'indique dans le sous-plot
        ax.text(0.5, 0.5, f"Fichier introuvable :\n{path}", 
                ha='center', va='center', fontsize=12)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    # Lecture du fichier .wav
    fs, signal = wavfile.read(path)

    # Si stéréo, on ne garde que le premier canal
    if signal.ndim > 1:
        signal = signal[:, 0]

    # Normalisation (si int16 → on ramène dans [-1,1])
    if signal.dtype == np.int16:
        signal = signal / 32768.0

    # Pour la FFT, on prend au maximum 65 536 échantillons (ou moins si le signal est plus court)
    N = min(65536, len(signal))
    snippet = signal[:N]

    # Calcul de la FFT et récupération de la moitié des fréquences positives
    fft_vals = np.fft.fft(snippet)
    mag     = np.abs(fft_vals)[:N//2]
    freqs   = np.fft.fftfreq(N, d=1/fs)[:N//2]

    # Tracé du spectre
    ax.plot(freqs, mag)
    ax.set_title(label, fontsize=14)
    ax.set_xlabel("Fréquence (Hz)")
    ax.xaxis.set_label_coords(0.81, -0.08)  
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, fs/2)
    ax.grid(True)

plt.tight_layout()
plt.show()

# Q3.4
# 3.4.1


def generer_signal(N, fe, f0, f1, A0, A1):
    t = np.arange(N) / fe  # vecteur temps en secondes
    x = A0 * np.cos(2 * np.pi * f0 * t) + A1 * np.cos(2 * np.pi * f1 * t)
    return x

N = 1000     # nombre d'échantillons
fe = 1000    # fréquence d'échantillonnage (Hz)
f0 = 50      # fréquence 1 (Hz)
f1 = 120     # fréquence 2 (Hz)
A0 = 1.0     # amplitude 1
A1 = 0.5     # amplitude 2

# Génération du signal
x = generer_signal(N, fe, f0, f1, A0, A1)

# Affichage
t = np.arange(N) / fe
plt.plot(t, x)
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Signal échantillonné")
plt.grid(True)
plt.show()


def verifier_hanning_fft(N):
    # Fenêtre de Hanning
    w = hann(N, sym=False)

    # Calcul de la FFT centrée et normalisée
    W = fft(w, 4096)  # on augmente la résolution FFT
    W = np.abs(fftshift(W))  # module + centrage

    # Axe fréquentiel normalisé [-0.5, 0.5]
    f = fftshift(fftfreq(len(W), d=1))  # pas d=1 car normalisé

    # Normalisation pour surface unité (approximation)
    W /= np.sum(W)

    # Tracé
    plt.figure(figsize=(8, 4))
    plt.plot(f, W, label="|FFT(Hanning)|")
    plt.title(f"FFT d'une fenêtre de Hanning de longueur N={N}")
    plt.xlabel("Fréquence normalisée")
    plt.ylabel("Amplitude normalisée")
    plt.grid(True)
    plt.xlim(-0.05, 0.05)
    plt.axvline(-2/N, color='r', linestyle='--', label='-2/N')
    plt.axvline(2/N, color='r', linestyle='--', label='2/N')
    plt.legend()
    plt.show()

# Exemple avec N = 64
verifier_hanning_fft(64)

def analyse_tranche_tfd(signal, fe):
    N = len(signal)
    window = hann(N)
    windowed_signal = signal * window
    X = np.abs(fft(windowed_signal))
    freqs = fftfreq(N, 1/fe)

    # On ne garde que la moitié positive du spectre
    half = slice(0, N//2)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs[half], X[half])
    plt.title("TFD de la tranche pondérée par une fenêtre de Hanning")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

# Paramètres du test
fe = 512_000  
f0 = 40_000   
f1 = 61_000   
A0 = A1 = 1
N = 256       

# Génération et analyse
signal = generer_signal(N, fe, f0, f1, A0, A1)
analyse_tranche_tfd(signal, fe)

# Q3.4.2

def fenetre(signal, type_fenetre):
    """Applique une fenêtre donnée au signal."""
    if type_fenetre == 'hanning':
        w = hann(len(signal))
    elif type_fenetre == 'hamming':
        w = hamming(len(signal))
    elif type_fenetre == 'blackman':
        w = blackman(len(signal))
    else:
        raise ValueError("Type de fenêtre non reconnu.")
    
    return signal * w

def analyser_spectre(subplot, signal, fe, fenetre_type):
    """Analyse le spectre du signal."""
    N = len(signal)
    X = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(N, 1/fe)
    half = slice(0, N//2)
    
    subplot.plot(freqs[half], X[half])
    subplot.set_title("Spectre pondéré par une fenêtre de " + fenetre_type)
    subplot.grid(True)
    subplot.set_xlabel("Fréquence (Hz)")
    subplot.set_ylabel("Amplitude")

# Paramètres du signal
N = 256     # nombre d'échantillons
fe = 8000   # fréquence d'échantillonnage (Hz)
f0 = 995    # fréquence 1 (Hz)
f1 = 1200   # fréquence 2 (Hz)
A0 = 1      # amplitude 1
A1 = 0.01   # amplitude 2

# Génération du signal
signal = generer_signal(N, fe, f0, f1, A0, A1)

fenetres = ['hanning', 'hamming', 'blackman']

fig, axs = plt.subplots(3, figsize=(8, 12))

for i, fenetre_type in enumerate(fenetres):
    signal_fenetre = fenetre(signal, fenetre_type)
    analyser_spectre(axs[i], signal_fenetre, fe, fenetre_type)

plt.tight_layout()
plt.show()