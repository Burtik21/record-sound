from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from pydub.utils import which
import os
import librosa
import numpy as np

# Nastavení ffmpeg ručně (Windows fix)
AudioSegment.converter = which("ffmpeg")

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
CONVERTED_FOLDER = "converted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "audio_data" not in request.files:
        return "No audio data", 400

    audio_file = request.files["audio_data"]
    filename = secure_filename(audio_file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(CONVERTED_FOLDER, filename.rsplit(".", 1)[0] + ".wav")

    # Ulož originální .webm
    audio_file.save(input_path)

    # Převod na WAV
    audio = AudioSegment.from_file(input_path, format="webm")
    audio.export(output_path, format="wav")

    # Analýza pomocí librosa
    y, sr = librosa.load(output_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist()
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    return jsonify({
        "mfcc": mfcc,
        "rms": rms,
        "zcr": zcr
    })

if __name__ == "__main__":
    app.run(debug=True)
