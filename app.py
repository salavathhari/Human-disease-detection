import os
import sqlite3
import traceback
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this!

# Flask-Login setup
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Database setup for users
DB_FILE = 'users.db'

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        ''')
        conn.commit()

init_db()

class User(UserMixin):
    def __init__(self, id_, username, password):
        self.id = id_
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = c.fetchone()
    if row:
        return User(row[0], row[1], row[2])
    return None

# Load image classification model
SKIN_MODEL_PATH = os.path.join("model", "skin2.keras")
if not os.path.exists(SKIN_MODEL_PATH):
    raise FileNotFoundError(f"Skin model file not found at {SKIN_MODEL_PATH}")
skin_model = load_model(SKIN_MODEL_PATH)
HEIGHT, WIDTH = skin_model.input_shape[1], skin_model.input_shape[2]

disease_labels = [
    'Eczema',
    'Warts Molluscum',
    'Atopic Dermatitis',
    'Basal Cell Carcinoma',
    'Melanocytic Nevi'
]

# Load symptom prediction model and precautions
SYMPTOM_MODEL_PATH = 'disease_precaution_model.pkl'
if not os.path.exists(SYMPTOM_MODEL_PATH):
    raise FileNotFoundError(f"Symptom model file not found at {SYMPTOM_MODEL_PATH}")

loaded_obj = joblib.load(SYMPTOM_MODEL_PATH)
symptom_model = loaded_obj.get('model')
precautions_map = loaded_obj.get('precautions')

if symptom_model is None or precautions_map is None:
    raise ValueError("Symptom model or precautions missing in loaded object")

# Folder for uploaded images
STATIC_UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(STATIC_UPLOAD_FOLDER, exist_ok=True)

# ---------- Routes ----------

@app.route('/')
@login_required
def root():
    return redirect(url_for('home'))

@app.route('/home')
@login_required
def home():
    return render_template('index.html')

@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = c.fetchone()

        if user and check_password_hash(user[2], password):
            login_user(User(user[0], user[1], user[2]))
            flash("Logged in successfully!")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password.")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not password or not confirm_password:
            flash("Please fill all the fields.")
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)

        try:
            with sqlite3.connect(DB_FILE) as conn:
                c = conn.cursor()
                c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
                conn.commit()
            flash("Signup successful! Please log in.")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists.")
            return redirect(url_for('signup'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.")
    return redirect(url_for('login'))

@app.route('/predict_image', methods=['POST'])
@login_required
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'})

        filename = secure_filename(file.filename)
        filepath = os.path.join(STATIC_UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(HEIGHT, WIDTH))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = skin_model.predict(img_array)
        if preds.shape[1] != len(disease_labels):
            return jsonify({'error': 'Prediction output size mismatch.'})

        confidence = float(np.max(preds)) * 100
        predicted_index = np.argmax(preds)

        THRESHOLD = 20
        if confidence < THRESHOLD:
            predicted_class = "No disease detected"
        else:
            predicted_class = disease_labels[predicted_index]

        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'image_url': url_for('static', filename=f'uploads/{filename}')
        })

    except Exception:
        tb = traceback.format_exc()
        print("Error during image prediction:", tb)
        return jsonify({'error': 'Internal error during image prediction', 'details': tb})

@app.route('/predict_symptoms', methods=['POST'])
@login_required
def predict_symptoms():
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'})

        symptoms_text = data['symptoms']
        symptoms_list = [s.strip().lower() for s in symptoms_text.split(',') if s.strip()]
        if not symptoms_list:
            return jsonify({'error': 'Empty symptoms list'})

        if not hasattr(symptom_model, 'feature_names_in_'):
            return jsonify({'error': 'Symptom model missing feature names'})

        feature_names = symptom_model.feature_names_in_

        input_vector = np.zeros(len(feature_names))
        for i, feature in enumerate(feature_names):
            if feature.lower() in symptoms_list:
                input_vector[i] = 1

        input_vector = input_vector.reshape(1, -1)
        disease_pred = symptom_model.predict(input_vector)[0]

        precautions = precautions_map.get(disease_pred, "No precautions available.")

        return jsonify({
            'disease': disease_pred,
            'precautions': precautions
        })

    except Exception:
        tb = traceback.format_exc()
        print("Error during symptom prediction:", tb)
        return jsonify({'error': 'Internal error during symptom prediction', 'details': tb})

if __name__ == '__main__':
    app.run(debug=True)
