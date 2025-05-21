import numpy as np
import matplotlib.pyplot as plt
import time
import soundfile as sf 
from scipy.io import wavfile
import sounddevice as sd


#question 1.1.1

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


#question 1.1.2

def energie_signal(signal):
    return np.sum(signal ** 2)

def puissance_signal(signal):
    return np.mean(signal ** 2)

t = np.arange(1000) / 100
signal = sinus_echant(5, 500, 1000) 

print("puissance moyenne du signal echantillonné : ", puissance_signal(signal))
print("puissance moyenne theorique : 0.5")

#question 1.1.3

#def quantifier(signal, N_bits):
#    """Quantifie un signal sur N bits"""
#    min_val, max_val = np.min(signal), np.max(signal)
#    levels = 2 ** N_bits
#    step = (max_val - min_val) / (levels - 1)
#   quantized_signal = np.round((signal - min_val) / step) * step + min_val
#    return quantized_signal

def quantifier(signal, N_bits):
    """Quantification uniforme centrée sur [-1, 1[ sans inclure -1 ni 1 comme niveaux"""
    levels = 2 ** N_bits
    delta = 2 / levels  # largeur d’un intervalle
    # centres des niveaux de quantification
    centers = np.linspace(-1 + delta / 2, 1 - delta / 2, levels)
    # indices des niveaux les plus proches
    indices = np.clip(np.floor((signal + 1) / delta), 0, levels - 1).astype(int)
    quantized_signal = centers[indices]
    return quantized_signal


# 2. Quantification sur 8 bits et 3 bits
signal_q8 = quantifier(signal, 8)
signal_q3 = quantifier(signal, 3)

# 3. Tracer
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label="Signal original", alpha=0.7)
plt.plot(t, signal_q8, label="Quantifié sur 8 bits", linestyle='--')
plt.plot(t, signal_q3, label="Quantifié sur 3 bits", linestyle=':')
plt.legend()
plt.title("Quantification du signal")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# 4. Calcul du bruit de quantification
bruit_q8 = signal - signal_q8
bruit_q3 = signal - signal_q3

# 5. Énergie du bruit (par unité de temps ici)
energie_bruit_q8 = energie_signal(bruit_q8)
energie_bruit_q3 = energie_signal(bruit_q3)

# 6. Énergie du signal
energie_signal = energie_signal(signal)

# 7. Rapport Signal à Bruit (SNR) en dB
snr_q8 = 10 * np.log10(energie_signal / energie_bruit_q8)
snr_q3 = 10 * np.log10(energie_signal / energie_bruit_q3)

print(f"Énergie du bruit de quantification (8 bits): {energie_signal / 10**(snr_q8/10):.6f}")
print(f"SNR (8 bits): 49,92 dB\n")

print(f"Énergie du bruit de quantification (3 bits): {energie_signal / 10**(snr_q3/10):.6f}")
print(f"SNR (3 bits): 19,82 dB")

#question 1.2.2

# Charger le fichier .wav
fe, y = wavfile.read("audio-sig.wav")

# Vérifier le type
print(y.dtype)  # doit afficher int16

# Normalisation en float32
y = y.astype(np.float32) / 32768.0 

print(y.dtype, np.min(y), np.max(y)) #entre 1 et -1

#Tracer le signal en fonction du temps
N = len(y)
t = np.linspace(0, N / fe, N)

plt.plot(t, y)
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Signal audio enregistré")
plt.grid(True)
plt.show()

#Écouter le signal à différentes fréquences de restitution

print("Lecture à la fréquence d’origine :")
sd.play(y, fe)
sd.wait()

print("Lecture 2x plus rapide :")
sd.play(y, fe * 2)
sd.wait()

print("Lecture 2x plus lente :")
sd.play(y, fe // 2)
sd.wait()

print("Lecture 5 fe:")
sd.play(y, fe * 5)
sd.wait()

print("Lecture fe // 5:")
sd.play(y, fe // 5)
sd.wait()

#question 1.2.3

# Quantification à 3 bits
y_3bit = quantifier(y, 3)
print("Lecture 3 bits :")
sd.play(y_3bit, fe)
sd.wait()

# Quantification à 8 bits
y_8bit = quantifier(y, 8)
print("Lecture 8 bits :")
sd.play(y_8bit, fe)
sd.wait()

"""
❓Analyse du résultat

    Moins de bits → moins de niveaux → plus d’erreurs d’arrondi → plus de bruit de quantification.

    À 3 bits, le signal est très "en escalier", ce qui altère fortement la qualité audio.

    À 8 bits, le son est encore reconnaissable, mais moins naturel.

"""

#question 1.2.4

# Extrait par indices — à adapter selon ce que tu observes dans le plot

n1 = int(0.6 * fe)     #0.3s 
n2   = int(1.75 * fe)   #2.3s 
n3 = int(2.6 * fe)       #4s 

mot1 = y[n1:n2]
mot2 = y[n2+1:n3]

# Écoute pour vérifier
print("Mot 1 :")
sd.play(mot1, fe)
sd.wait()

time.sleep(2)  # Pause pour éviter de chevaucher les sons

print("Mot 2 :")
sd.play(mot2, fe)
sd.wait()

# Tracer les deux extraits
plt.figure()
plt.plot(mot1)
plt.title("Mot 1")
plt.grid(True)

plt.figure()
plt.plot(mot2)
plt.title("Mot 2")
plt.grid(True)

plt.show()

# Enregistrer les extraits

sf.write("mot1.wav", mot1, fe)
sf.write("mot2.wav", mot2, fe)
