import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

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

def quantifier(signal, N_bits):
    """Quantifie un signal sur N bits"""
    min_val, max_val = np.min(signal), np.max(signal)
    levels = 2 ** N_bits
    step = (max_val - min_val) / (levels - 1)
    quantized_signal = np.round((signal - min_val) / step) * step + min_val
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

print(f"Énergie du bruit de quantification (8 bits): {energie_bruit_q8:.6f}")
print(f"SNR (8 bits): {snr_q8:.2f} dB\n")

print(f"Énergie du bruit de quantification (3 bits): {energie_bruit_q3:.6f}")
print(f"SNR (3 bits): {snr_q3:.2f} dB")

#question 1.2.1