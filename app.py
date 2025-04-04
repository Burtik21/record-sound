import librosa
from flask import Flask, request
import os
import soundfile as sf
from audio.audio_utils import AudioProcessor
from audio.analyze_wav import detect_segments_and_extract_features

app = Flask(__name__)


@app.route('/process_audio', methods=['POST'])
def process_audio():
    # Získání cesty k souboru
    file_path = request.json.get('filePath')
    processor = AudioProcessor()

    # Kontrola, zda soubor existuje
    if not file_path or not os.path.exists(file_path):
        return {"error": "Soubor nenalezen."}, 400

    print(f"Kontroluji existenci souboru: {file_path}")
    print("Soubor existuje.")

    try:

        clicks = detect_segments_and_extract_features(file_path)
        return {"clicks": clicks}, 200
    except Exception as e:
        print(f"Chyba při čtení souboru: {e}")
        return {"error": "Chyba při zpracování souboru."}, 500




if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5499)
