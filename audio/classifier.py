import pandas as pd
import pickle
input_features = [
    'rms', 'zcr', 'spectral_centroid', 'spectral_bandwidth',
    'spectral_rolloff', 'max_intensity', 'dominant_freq',
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_6',
    'mfcc_7', 'mfcc_8', 'mfcc_13','duration'
]
with open('audio/model.pkl', 'rb') as f:
    model = pickle.load(f)


def classify(features):
    """
    Uloží seznam features do CSV souboru.

    :param features: Seznam slovníků obsahujících různé vlastnosti (features)
    :param csv_filename: Název CSV souboru pro uložení dat
    """
    # Převeď seznam features na DataFrame
    df = pd.DataFrame([features])
    x_input = df[input_features]
    pred = int(model.predict(x_input)[0])
    return pred
