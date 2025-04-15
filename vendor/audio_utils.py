
import numpy as np
import librosa
import librosa.display
import scipy.signal as signal
import noisereduce as nr
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os


class AudioProcessor:
    def __init__(self, rate=44100):
        self.rate = rate

    def highpass_filter(self, data, cutoff=300):
        nyquist = 0.5 * self.rate
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(1, normal_cutoff, btype='high')
        return signal.filtfilt(b, a, data)

    def bandpass_filter(self, data, lowcut=300, highcut=8000):
        nyquist = 0.5 * self.rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(2, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def apply_filters(self, audio):
        audio_hp = self.highpass_filter(audio)
        audio_bp = self.bandpass_filter(audio_hp)
        return audio_bp


    def extract_features(self, y):
        features = {
            'rms': float(np.sqrt(np.mean(y ** 2))),
            'zcr': float(np.mean(librosa.feature.zero_crossing_rate(y))),
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=self.rate))),
            'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=self.rate))),
            'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=self.rate))),
            'max_intensity': float(np.max(np.abs(y))),
            'dominant_freq': float(np.fft.rfftfreq(len(y), 1 / self.rate)[np.argmax(np.abs(np.fft.rfft(y)))])
        }

        # MFCCs (s vynecháním vybraných indexů)
        mfccs = librosa.feature.mfcc(y=y, sr=self.rate, n_mfcc=13).mean(axis=1)
        skip = [5, 11, 10,12,9]  # tyto MFCC čísla nechceš
        for i, val in enumerate(mfccs):
            if (i + 1) in skip:
                continue
            features[f'mfcc_{i + 1}'] = float(val)

        return features

