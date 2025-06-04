import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import os


def is_voiced(frame, energy_threshold):
    """
    Détermine si une trame est voisée selon son énergie.
    """
    energy = np.sum(frame ** 2) / len(frame)
    return energy > energy_threshold

def autocorrelation_pitch(frame, fs, fmin, fmax):
    """
    Estime la fréquence fondamentale d'une trame via l'autocorrélation.
    Retourne None si la recherche échoue.
    """
    # Soustraire la moyenne pour centrer
    frame = frame - np.mean(frame)
    # Calculer l'autocorrélation
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr)//2:]  # ne garder que la partie positive
    # Déterminer les bornes de lag correspondant à fmin et fmax
    lag_min = int(fs / fmax)
    lag_max = int(fs / fmin)
    if lag_max >= len(corr):
        lag_max = len(corr) - 1
    # Chercher le pic dans la plage de lag
    segment = corr[lag_min:lag_max+1]
    if len(segment) == 0:
        return None
    peak_idx = np.argmax(segment) + lag_min
    if corr[peak_idx] <= 0:
        return None
    # Convertir le lag en fréquence
    freq = fs / peak_idx
    return freq

def freq_to_midi(freq):
    """
    Convertit une fréquence en numéro de note MIDI.
    Si freq est None ou <= 0, retourne None.
    """
    if freq is None or freq <= 0:
        return None
    return int(np.round(69 + 12 * np.log2(freq / 440.0)))

def detect_pitch_sequence(
    wav_filename,
    frame_duration=0.030,
    pitch_range=(80.0, 400.0),
    min_note_duration=0.1
):
    """
    Fonction principale de détection de pitch et de notes.

    Paramètres :
    - wav_filename : chemin vers le fichier .wav
    - frame_duration : durée (en secondes) de chaque trame
    - pitch_range : (fmin, fmax) domaine de recherche du pitch en Hz
    - min_note_duration : durée minimale d'une note en secondes

    Retourne :
    - matrice numpy de shape (N, 3) avec colonnes [temps, fréquence, note_MIDI]
    """
    # Lecture du signal
    fs, signal_data = wav.read(wav_filename)

    # Si le signal est stéréo, le convertir en mono
    if signal_data.ndim == 2:
        signal_data = signal_data.mean(axis=1)

    # Normaliser le signal entre -1 et 1 si nécessaire
    if signal_data.dtype != np.float32 and signal_data.dtype != np.float64:
        max_val = np.iinfo(signal_data.dtype).max
        signal_data = signal_data.astype(np.float32) / max_val

    # Paramètres de tramage
    frame_len = int(frame_duration * fs)
    hop_len = frame_len  # trames sans recouvrement
    num_frames = int(np.floor(len(signal_data) / hop_len))

    # Calcul du seuil d'énergie (par exemple, 10% de l'énergie max d'une trame)
    energies = []
    for i in range(num_frames):
        start = i * hop_len
        frame = signal_data[start : start + frame_len]
        energies.append(np.sum(frame ** 2) / len(frame))
    energy_threshold = 0.05 * np.max(energies)

    # Allocation des résultats
    times = np.zeros(num_frames)
    freqs = np.zeros(num_frames)
    midis = np.zeros(num_frames, dtype=int)

    fmin, fmax = pitch_range

    # Parcours des trames
    for i in range(num_frames):
        start = i * hop_len
        frame = signal_data[start : start + frame_len]
        times[i] = (start + frame_len / 2) / fs  # temps au centre de la trame

        if is_voiced(frame, energy_threshold):
            freq = autocorrelation_pitch(frame, fs, fmin, fmax)
            freqs[i] = freq if freq is not None else 0.0
            midi = freq_to_midi(freq) if freq is not None else 0
            midis[i] = midi if midi is not None else 0
        else:
            freqs[i] = 0.0
            midis[i] = 0

    # Détection des notes : regrouper les trames de même note MIDI consécutives
    min_frames_per_note = int(np.ceil(min_note_duration / frame_duration))
    i = 0
    while i < num_frames:
        if midis[i] > 0:
            j = i + 1
            while j < num_frames and midis[j] == midis[i]:
                j += 1
            length = j - i
            if length < min_frames_per_note:
                # Pas assez long : considérer comme bruit -> zéro
                freqs[i:j] = 0.0
                midis[i:j] = 0
            i = j
        else:
            i += 1

    # Construire la matrice résultat
    result = np.column_stack((times, freqs, midis))
    return result

# Utilisation
for file in os.listdir("./fichierzip/"):
    if file.endswith(".wav") or file.endswith(".WAV"):
        print(f"Traitement du fichier : {file}")

        result = detect_pitch_sequence(
            f"./fichierzip/{file}",
            frame_duration=0.025,
            pitch_range=(80.0, 300.0),
            min_note_duration=0.1
        )
        np.savetxt(f"./partie4_output/pitch_detection_{file[:-4]}.csv", result, delimiter=",",
                header="time,frequency,midi", comments='')
