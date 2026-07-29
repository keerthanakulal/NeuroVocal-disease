import sys
import os
import threading
import webbrowser
from flask import Flask, request, render_template, jsonify, redirect, url_for, session
import numpy as np
import librosa
import joblib
import onnxruntime as ort
import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment
from io import BytesIO
import logging
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import json
from datetime import datetime

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
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        crnn_model = ort.InferenceSession(os.path.join(MODEL_DIR, 'crnn_model.onnx'), sess_options=opts)
        rf_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.joblib'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'audio_scaler.joblib'))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
        rf_feature_cols = joblib.load(os.path.join(MODEL_DIR, 'rf_feature_columns.joblib'))
        
        models_loaded = True
        app.logger.info("✅ All models loaded successfully!")

    except Exception as e:
        app.logger.critical(f"❌ FATAL ERROR: Could not load models. Error: {e}", exc_info=True)

# Database Configuration
db_url = os.environ.get('DATABASE_URL')
if db_url:
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'database.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'neurovocal-diagnostics-portal-session-secret-key-9988'

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    scans = db.relationship('Scan', backref='user', lazy=True, cascade="all, delete-orphan")

class Scan(db.Model):
    __tablename__ = 'scans'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.String(50), nullable=False)
    label = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    probs_json = db.Column(db.Text, nullable=False)

# Auto-create tables & default user
with app.app_context():
    db.create_all()
    default_email = "doctor@hospital.com"
    if not User.query.filter_by(email=default_email).first():
        hashed = bcrypt.hashpw("password123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        default_user = User(email=default_email, password_hash=hashed, name="Dr. Smith")
        db.session.add(default_user)
        db.session.commit()

@app.route('/')
def index():
    """Redirects to dashboard if logged in, otherwise to login."""
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_page'))

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            session['user'] = email
            session['username'] = user.name
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Invalid email or password.'}), 401
    
    # GET request
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    
    if not name or not email or not password:
        return jsonify({'success': False, 'error': 'All fields are required.'}), 400
        
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'error': 'An account with this email already exists.'}), 400
        
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = User(email=email, password_hash=hashed, name=name)
    db.session.add(new_user)
    db.session.commit()
    
    session['user'] = email
    session['username'] = name
    return jsonify({'success': True})

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('username', None)
    return redirect(url_for('login_page'))

@app.route('/api/user')
def get_user():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user = User.query.filter_by(email=session['user']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({
        'email': user.email,
        'username': user.name
    })

@app.route('/api/history')
def get_history():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user = User.query.filter_by(email=session['user']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    scans = Scan.query.filter_by(user_id=user.id).order_by(Scan.id.desc()).limit(20).all()
    history_list = []
    for s in scans:
        history_list.append({
            'date': s.date,
            'label': s.label,
            'confidence': round(s.confidence * 100, 1) if s.confidence <= 1.0 else round(s.confidence, 1)
        })
    return jsonify(history_list)

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
        
        X_crnn = np.expand_dims(mel_spec_db, axis=0).astype(np.float32)
        ort_inputs = {crnn_model.get_inputs()[0].name: X_crnn}
        ort_outs = crnn_model.run(None, ort_inputs)
        crnn_probs = ort_outs[0][0]

        # --- Ensemble and Final Result ---
        ensemble_probs = (rf_probs + crnn_probs) / 2.0
        pred_idx = int(np.argmax(ensemble_probs))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(ensemble_probs[pred_idx])
        
        class_labels = label_encoder.classes_.tolist()
        probs_dict = {class_labels[i]: float(ensemble_probs[i]) for i in range(len(class_labels))}

        # --- Save prediction result to Database if user is logged in ---
        if 'user' in session:
            user = User.query.filter_by(email=session['user']).first()
            if user:
                date_str = datetime.now().strftime("%m/%d/%Y %I:%M %p")
                scan = Scan(
                    user_id=user.id,
                    date=date_str,
                    label=pred_label,
                    confidence=confidence,
                    probs_json=json.dumps(probs_dict)
                )
                db.session.add(scan)
                db.session.commit()
        
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