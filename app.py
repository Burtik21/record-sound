import librosa
from flask import Flask, request
import os
import soundfile as sf
from audio.audio_utils import AudioProcessor

app = Flask(__name__)


@app.route('/process_audio', methods=['POST'])
def process_audio():
    # Získání cesty k souboru
    file_path = request.json.get('filePath')
    processor = AudioProcessor()
    data = []

    # Kontrola, zda soubor existuje
    if not file_path or not os.path.exists(file_path):
        return {"error": "Soubor nenalezen."}, 400

    print(f"Kontroluji existenci souboru: {file_path}")
    print("Soubor existuje.")

    try:
        # Načtení souboru
        y, sr = sf.read(file_path)

        # Aplikování filtrů
        filtered_audio = processor.apply_filters(y)

        # Detekce impulzů
        impulses = processor.detect_impulses(filtered_audio, threshold=0.0011)

        if impulses:
            start, end, duration = impulses[0]  # vezmeme jen první
            print(f"✅ Impulz: začátek={start:.4f}s, konec={end:.4f}s, trvání={duration:.4f}s")

            impulse_audio = filtered_audio[int(start * sr):int(end * sr)]
            features = processor.extract_features(impulse_audio)
            features['duration'] = duration
            features['sound_category'] = '1'  # nebo 'klavesnice', jak chceš

            data.append(features)
            print("✅ Features uloženy.")

            # Pokusíme se soubor smazat až po zpracování
            try:
                os.remove(file_path)
                print(f"Soubor {file_path} byl úspěšně smazán.")
            except FileNotFoundError:
                print(f"Soubor {file_path} nebyl nalezen pro smazání.")
            except Exception as e:
                print(f"Došlo k chybě při mazání souboru: {e}")

            return {
                "message": "Soubor zpracován",
                "features": data
            }, 200
        else:
            print("⚠️ Žádný impulz nenalezen. Vzorek přeskočen.")

            # Pokusíme se soubor smazat, pokud impulzy nebyly nalezeny
            try:
                os.remove(file_path)
                print(f"Soubor {file_path} byl úspěšně smazán.")
            except FileNotFoundError:
                print(f"Soubor {file_path} nebyl nalezen pro smazání.")
            except Exception as e:
                print(f"Došlo k chybě při mazání souboru: {e}")

    except Exception as e:
        print(f"Chyba při čtení souboru: {e}")
        return {"error": "Chyba při zpracování souboru."}, 500

    # Pokud nebyl nalezen impuls
    return {
        "message": "Nic nenalezeno",
    }, 200


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5499)
