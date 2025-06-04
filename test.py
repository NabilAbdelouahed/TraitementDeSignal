import os

for file in os.listdir("./fichierzip/"):
    if file.endswith(".wav") or file.endswith(".WAV"):
        print(f"Traitement du fichier : {file[:-4]}")