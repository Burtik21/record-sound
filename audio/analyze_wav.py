import soundfile as sf
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from audio.audio_utils import AudioProcessor  # 👈 Uprav podle cesty ke třídě
from audio.classifier import classify

def detect_segments_and_extract_features(audio_file,threshold=0.005, cooldown_time=0.12):
    processor = AudioProcessor()
    y, sr = sf.read(audio_file)
    filtered_audio = processor.apply_filters(y)
    hop_length = 256
    frame_length = 1024

    # Výpočet RMS pro zvuková data
    rms = librosa.feature.rms(y=filtered_audio, frame_length=frame_length,hop_length=hop_length)[0]
    times_rms = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

    # Parametry pro CSV soubor
    data = []
    start = None
    clicks = 0
    last_end_time = -cooldown_time  # Udržuj čas posledního segmentu, aby byl cooldown zajištěn

    for i in range(1, len(rms)):
        if rms[i - 1] < threshold and rms[i] >= threshold:
            # Začátek segmentu (když RMS přesáhne práh)
            start = i
        elif rms[i - 1] >= threshold and rms[i] < threshold and start is not None:
            # Konec segmentu (když RMS klesne pod práh)
            end = i
            segment_start_time = times_rms[start]
            segment_end_time = times_rms[end]

            # Pokud mezi segmenty není dostatek času (cooldown), ignorujeme tento segment
            if segment_start_time - last_end_time >= cooldown_time:

                # Extrakce vlastností pro daný segment
                duration = segment_end_time - segment_start_time
                segment_audio = y[start * hop_length:end * hop_length]  # Získej zvukový segment
                features = processor.extract_features(segment_audio)  # Extrahuj vlastnosti
                features['duration'] = duration

                pred = classify(features)
                if pred == 1:
                    clicks += 1
                else:
                    print("zadne clicks")

                last_end_time = segment_end_time

            start = None

    return clicks


