import sys
import os
import threading
import webbrowser
from flask import Flask, request, render_template, jsonify
import numpy as np
import librosa
import joblib
import tensorflow as tf
import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment
from io import BytesIO
import logging

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Initialize the Flask app
app = Flask(__name__, 
           template_folder=resource_path('templates'),
           static_folder=resource_path('static'))

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Globals for Models ---
crnn_model = None
rf_model = None
scaler = None
label_encoder = None
rf_feature_cols = None
models_loaded = False

def load_models():
    global crnn_model, rf_model, scaler, label_encoder, rf_feature_cols, models_loaded
    try:
        # Use resource_path to find models directory
        MODEL_DIR = resource_path('models')
        app.logger.info(f"Attempting to load models from the '{MODEL_DIR}' directory...")
        
        # Load all the necessary model and pre-processing files
        crnn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'crnn_model.h5'))
        rf_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.joblib'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'audio_scaler.joblib'))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
        rf_feature_cols = joblib.load(os.path.join(MODEL_DIR, 'rf_feature_columns.joblib'))
        
        models_loaded = True
        app.logger.info("✅ All models loaded successfully!")

    except Exception as e:
        app.logger.critical(f"❌ FATAL ERROR: Could not load models. Error: {e}", exc_info=True)

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the audio prediction logic."""
    if not models_loaded:
        app.logger.error("A prediction request was received, but the models are not loaded.")
        return jsonify({'error': 'Models are not loaded on the server. Please check server logs.'}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found in the request.'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    import tempfile
    
    # Save the uploaded file to a temporary file
    temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_wav_fd)
    
    try:
        # --- Audio Transcoding using pydub ---
        audio_buffer = BytesIO(file.read())
        sound = AudioSegment.from_file(audio_buffer)
        sound.export(temp_wav_path, format='wav')

        # --- Feature Extraction ---
        y, sr = librosa.load(temp_wav_path, sr=44100)
        sound_obj = parselmouth.Sound(temp_wav_path)
        
        features = {}
        try:
            pp = call(sound_obj, "To PointProcess (periodic, cc)", 75, 500)
            features['jitter_local'] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features['shimmer_local'] = call([sound_obj, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            harmonicity = call(sound_obj, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            features['hnr'] = call(harmonicity, "Get mean", 0, 0)
            pitch = sound_obj.to_pitch(None, 75, 600)
            intensity = sound_obj.to_intensity()
            features['mean_f0'] = call(pitch, "Get mean", 0, 0, "Hertz")
            features['std_dev_f0'] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            features['mean_intensity'] = call(intensity, "Get mean", 0, 0, "energy")
        except Exception as praat_error:
            app.logger.warning(f"Praat feature extraction failed: {praat_error}. Defaulting acoustic features to 0.")
            features.update({'jitter_local':0, 'shimmer_local':0, 'hnr':0, 'mean_f0':0, 'std_dev_f0':0, 'mean_intensity':0})
            
        # Extract Librosa features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Ensure all required features are present
        for col in rf_feature_cols:
            features.setdefault(col, 0)
        
        # --- Prediction ---
        X_rf = scaler.transform([np.array([features[col] for col in rf_feature_cols])])
        rf_probs = rf_model.predict_proba(X_rf)[0]

        # CRNN Prediction
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        max_len = 250
        if mel_spec_db.shape[1] > max_len:
            mel_spec_db = mel_spec_db[:, :max_len]
        else:
            pad_width = max_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0,0),(0,pad_width)), mode='constant')
        
        X_crnn = np.expand_dims(mel_spec_db, axis=(0, 3))
        crnn_probs = crnn_model.predict(X_crnn, verbose=0)[0]

        # --- Ensemble and Final Result ---
        ensemble_probs = (rf_probs + crnn_probs) / 2.0
        pred_idx = int(np.argmax(ensemble_probs))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(ensemble_probs[pred_idx])
        
        class_labels = label_encoder.classes_.tolist()
        probs_dict = {class_labels[i]: float(ensemble_probs[i]) for i in range(len(class_labels))}
        
        return jsonify({
            'label': pred_label,
            'confidence': confidence,
            'probs': probs_dict
        })

    except Exception as e:
        app.logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred while processing the audio file.'}), 500
    finally:
        # Clean up temporary WAV file
        try:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception as cleanup_error:
            app.logger.warning(f"Failed to clean up temporary file {temp_wav_path}: {cleanup_error}")

def open_browser():
    """Open web browser after a delay"""
    import time
    time.sleep(1.5)  # Wait for Flask to start
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Load models first
    load_models()
    
    # Check if running in a cloud deployment environment (where PORT is injected)
    is_cloud = 'PORT' in os.environ
    
    if not is_cloud:
        # Open browser in a separate thread
        threading.Timer(1, open_browser).start()
        host = '127.0.0.1'
        port = 5000
    else:
        host = '0.0.0.0'
        port = int(os.environ.get('PORT', 5000))
        app.logger.info(f"Cloud deployment detected. Binding to {host}:{port}")
    
    # Run Flask app
    app.run(host=host, port=port, debug=False, use_reloader=False)