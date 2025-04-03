
import numpy as np
import librosa
import librosa.display
import scipy.signal as signal
import noisereduce as nr
import matplotlib.pyplot as plt


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

    def plot_rms(self, audio, threshold=0.0015):
        rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
        times = librosa.times_like(rms, sr=self.rate, hop_length=256)

        plt.figure(figsize=(12, 4))
        plt.plot(times, rms, label='RMS', color='orange')
        plt.axhline(threshold, color='red', linestyle='--')
        plt.xlabel('Čas (s)')
        plt.ylabel('RMS')
        plt.title('RMS energie s prahem')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_spectrogram(self, audio):
        plt.figure(figsize=(12, 5))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=self.rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spektrogram')
        plt.xlabel('Čas (s)')
        plt.ylabel('Frekvence (Hz)')
        plt.show()

    def detect_impulses(self, audio, threshold=0.0015, min_duration=0.005):
        rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
        above_thresh = rms > threshold
        impulses = np.diff(above_thresh.astype(int))
        starts = np.where(impulses == 1)[0]
        ends = np.where(impulses == -1)[0]

        impulse_list = []

        if len(starts) == 0 or len(ends) == 0:
            return []

        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(ends) < len(starts):
            starts = starts[:len(ends)]

        for s, e in zip(starts, ends):
            start_time = librosa.frames_to_time(s, sr=self.rate, hop_length=256)
            end_time = librosa.frames_to_time(e, sr=self.rate, hop_length=256)
            duration = end_time - start_time
            if duration >= min_duration:
                impulse_list.append((start_time, end_time, duration))

        return impulse_list

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

