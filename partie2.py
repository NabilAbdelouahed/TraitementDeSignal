import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd
from scipy.io import wavfile
import librosa
import soundfile as sf
from scipy.signal import correlate, find_peaks


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


#Q 2.3


# Paramètres communs
fe = 11000          # fréquence d'échantillonnage
duration = 1     # durée en secondes
t = np.linspace(0, duration, int(fe * duration), endpoint=False)

# 1. Signal sinusoïdal de 200 Hz
f = 200
sinus = np.sin(2 * np.pi * f * t)

# 2. Signal triangulaire centré en 0 (série de Fourier avec 10 termes)
tri = np.zeros_like(t)
for k in range(1, 11):
    n = 2 * k - 1  # uniquement les harmoniques impaires
    tri += ((-1)**((k+1)) / n**2) * np.sin(2 * np.pi * n * f * t)
tri *= (8 / (np.pi**2))  # normalisation série de Fourier

# 3. Bruit blanc gaussien
taille = fe  # 1 seconde
bruit = np.random.randn(taille)
bruit /= np.max(np.abs(bruit))  # normalisation pour éviter la saturation

# Sauvegarde du bruit en fichier audio
write("monbruit.wav", fe, bruit.astype(np.float32))


# Affichage
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, sinus)
plt.title("Signal sinusoïdal (200 Hz)")

plt.subplot(3, 1, 2)
plt.plot(t, tri)
plt.title("Signal triangulaire (200 Hz) - série de Fourier")

plt.subplot(3, 1, 3)
plt.plot(np.linspace(0, 1, taille), bruit)
plt.title("Bruit blanc gaussien (1 seconde)")

plt.tight_layout()
plt.show()



def autocorr(x):
    x = x - np.mean(x)
    result = correlate(x, x, mode='full')
    result /= np.max(result)  # Normalisation
    return result[result.size // 2:]

def slice_signal(signal, fe, start_ms, duration_ms):
    start = int(fe * start_ms / 1000)
    end = int(start + fe * duration_ms / 1000)
    return signal[start:end]

# Charger les signaux
fe_aa, aa = wavfile.read("a-syllabe.wav")
fe_bruit, bruit = wavfile.read("monbruit.wav")

# Tranches de signaux (30 ms)
aa_tranche1 = slice_signal(aa, fe_aa, 1000, 30)
aa_tranche2 = slice_signal(aa, fe_aa, 1500, 30)

bruit_tranche = slice_signal(bruit, fe, 0, 30)
sinus_tranche = slice_signal(sinus, fe, 0, 30)
tri_tranche = slice_signal(tri, fe, 0, 30)

# Autocorrélations
autocorrs = {
    "aa (1000 ms)": autocorr(aa_tranche1),
    "aa (1500 ms)": autocorr(aa_tranche2),
    "bruit": autocorr(bruit_tranche),
    "sinusoïde": autocorr(sinus_tranche),
    "triangulaire": autocorr(tri_tranche),
}

# Affichage
plt.figure(figsize=(12, 8))
for i, (label, ac) in enumerate(autocorrs.items(), 1):
    plt.subplot(3, 2, i)
    plt.plot(ac)
    plt.title(f"Autocorrélation - {label}")
    plt.grid(True)

plt.tight_layout()
plt.show()


# Q 2.4

fe_chat, chat = wavfile.read("chat_resample.wav")

def autocorr_sig(x):
    x = x - np.mean(x)
    r = correlate(x, x, mode='full')
    r = r[len(r)//2:]
    r /= np.max(r)
    return r

def extract_tranche(signal, fe, start_ms, duration_ms):
    start = int(start_ms * fe / 1000)
    end = start + int(duration_ms * fe / 1000)
    return signal[start:end]

# Exemple : analyser 3 tranches
tranches = {
    "chhh": extract_tranche(chat, fe_chat, 700, 30),
    "aaa": extract_tranche(chat, fe_chat, 2100, 30),
}


plt.figure(figsize=(12, 4))
for i, (label, tranche) in enumerate(tranches.items()):
    ac = autocorr_sig(tranche)
    plt.subplot(1, 2, i+1)
    plt.plot(ac)
    plt.title(f"Autocorrélation - {label}")
    plt.grid(True)

plt.tight_layout()
plt.show()


def freq_fondamentale(ac, fe):
    peaks, _ = find_peaks(ac, height=0.3)
    plt.plot(ac_a)
    plt.plot(peaks, ac_a[peaks], "x")
    plt.title("Autocorrélation de 'aaa' avec pics détectés")
    plt.grid(True)
    plt.show()
    if len(peaks) > 1:
        print(f"Pics détectés : {peaks[:4]}")
        delay = peaks[0]
        print(f"Premier délai : {delay} échantillons")
        print(delay)
        return fe / delay
    return None

# Normalisation
for key in tranches:
    tranches[key] = tranches[key] / np.max(np.abs(tranches[key]))

ac_a = autocorr(tranches["aaa"])
f0 = freq_fondamentale(ac_a, fe_chat)
print(f"Fréquence fondamentale estimée : {f0:.1f} Hz")



