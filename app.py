import os
os.environ["STREAMLIT_CONFIG_DIR"] = ".streamlit"
import sqlite3
import traceback
import numpy as np
import joblib
import streamlit as st
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ----------------- Database Setup -----------------
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

# ----------------- User Functions -----------------
def signup_user(username, password):
    hashed = generate_password_hash(password)
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        row = c.fetchone()
    if row and check_password_hash(row[2], password):
        return True
    return False

# ----------------- Load Models -----------------
SKIN_MODEL_PATH = os.path.join("model", "skin2.keras")
if not os.path.exists(SKIN_MODEL_PATH):
    st.error(f"Skin model file not found at {SKIN_MODEL_PATH}")
else:
    skin_model = load_model(SKIN_MODEL_PATH)
    HEIGHT, WIDTH = skin_model.input_shape[1], skin_model.input_shape[2]

disease_labels = [
    'Eczema',
    'Warts Molluscum',
    'Atopic Dermatitis',
    'Basal Cell Carcinoma',
    'Melanocytic Nevi'
]

SYMPTOM_MODEL_PATH = 'disease_precaution_model.pkl'
if os.path.exists(SYMPTOM_MODEL_PATH):
    loaded_obj = joblib.load(SYMPTOM_MODEL_PATH)
    symptom_model = loaded_obj.get('model')
    precautions_map = loaded_obj.get('precautions')
else:
    symptom_model = None
    precautions_map = {}

# ----------------- Helper Functions -----------------
def predict_skin_disease(uploaded_file):
    try:
        img = image.load_img(uploaded_file, target_size=(HEIGHT, WIDTH))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = skin_model.predict(img_array)
        confidence = float(np.max(preds)) * 100
        predicted_index = np.argmax(preds)

        THRESHOLD = 20
        if confidence < THRESHOLD:
            return "No disease detected", confidence
        else:
            return disease_labels[predicted_index], confidence
    except Exception:
        return "Error in prediction", 0

def predict_symptoms(symptoms_text):
    try:
        symptoms_list = [s.strip().lower() for s in symptoms_text.split(',') if s.strip()]
        if not symptoms_list:
            return "Error: Empty symptoms list", "No precautions"

        if not hasattr(symptom_model, 'feature_names_in_'):
            return "Error: Symptom model missing features", "No precautions"

        feature_names = symptom_model.feature_names_in_
        input_vector = np.zeros(len(feature_names))
        for i, feature in enumerate(feature_names):
            if feature.lower() in symptoms_list:
                input_vector[i] = 1

        input_vector = input_vector.reshape(1, -1)
        disease_pred = symptom_model.predict(input_vector)[0]
        precautions = precautions_map.get(disease_pred, "No precautions available.")

        return disease_pred, precautions
    except Exception as e:
        return f"Error: {str(e)}", "No precautions"

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Medical Assistant", layout="wide")

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

st.title("🩺 Medical Assistant")

# ----------- Login / Signup -----------
if not st.session_state.logged_in:
    choice = st.sidebar.radio("Menu", ["Login", "Signup"])

    if choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username}!")
            else:
                st.error("Invalid username or password")

    elif choice == "Signup":
        st.subheader("Create an Account")
        username = st.text_input("Choose a Username")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        if st.button("Signup"):
            if not username or not password:
                st.warning("Please fill all fields")
            elif password != confirm:
                st.error("Passwords do not match")
            else:
                if signup_user(username, password):
                    st.success("Signup successful! Please login.")
                else:
                    st.error("Username already exists.")

# ----------- Main App -----------
else:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()

    tab1, tab2, tab3 = st.tabs([
        "🖼 Skin Disease Prediction",
        "📝 Symptom Checker",
        "💬 Chatbot"
    ])

    with tab1:
        st.subheader("Upload a Skin Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if st.button("Predict Disease"):
                pred, conf = predict_skin_disease(uploaded_file)
                st.success(f"Prediction: {pred} (Confidence: {conf:.2f}%)")

    with tab2:
        st.subheader("Enter Symptoms (comma-separated)")
        symptoms_text = st.text_area("Example: itching, skin rash, nodal skin eruptions")
        if st.button("Predict from Symptoms"):
            disease, precautions = predict_symptoms(symptoms_text)
            st.success(f"Predicted Disease: {disease}")
            st.info(f"Precautions: {precautions}")

    with tab3:
        st.subheader("Chatbot 🤖")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("You:", key="chat_input")
        if st.button("Send"):
            if user_input:
                # Simple rule-based response for now
                if "hello" in user_input.lower():
                    bot_response = "Hi there! How can I help you today?"
                elif "symptom" in user_input.lower():
                    bot_response = "You can use the Symptom Checker tab to predict diseases."
                elif "skin" in user_input.lower():
                    bot_response = "Upload your skin image in the Skin Disease Prediction tab."
                else:
                    bot_response = "I'm just a simple chatbot here to assist you."

                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", bot_response))

        # Display chat history
        for sender, msg in st.session_state.chat_history:
            if sender == "You":
                st.markdown(f"**🧑 You:** {msg}")
            else:
                st.markdown(f"**🤖 Bot:** {msg}")
