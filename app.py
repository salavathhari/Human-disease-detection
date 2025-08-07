import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model, load_model
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA LOADING AND PROCESSING (Same as before)
# ============================================

def load_and_preprocess():
    """Same loading and preprocessing function as before"""
    # [Keep your existing implementation]
    return df

# ============================================
# MODEL TRAINING WITH KERAS/TENSORFLOW
# ============================================

def build_keras_models(X_train_tfidf, y_disease, y_precautions):
    """Build and train Keras models for disease and precaution prediction"""
    num_diseases = len(np.unique(y_disease))
    num_precautions = y_precautions.shape[1]
    
    # Disease classification model
    disease_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
        Dense(64, activation='relu'),
        Dense(num_diseases, activation='softmax')
    ])
    
    disease_model.compile(
        optimizer=Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Precaution prediction model (multi-label)
    precaution_model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
        Dense(128, activation='relu'),
        Dense(num_precautions, activation='sigmoid')
    ])
    
    precaution_model.compile(
        optimizer=Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return disease_model, precaution_model

def train_keras_models(df):
    """Train Keras models and return with encoders"""
    # Prepare data
    X = df['symptoms_text']
    
    # Disease encoder
    disease_encoder = LabelEncoder()
    y_disease = disease_encoder.fit_transform(df['disease_clean'])
    
    # Precaution encoder
    mlb = MultiLabelBinarizer()
    y_precautions = mlb.fit_transform(df['precautions_list'])
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_features=1000
    )
    X_tfidf = vectorizer.fit_transform(X).toarray()
    
    # Split data
    X_train, X_test, y_d_train, y_d_test, y_p_train, y_p_test = train_test_split(
        X_tfidf, y_disease, y_precautions, test_size=0.2, random_state=42)
    
    # Build models
    disease_model, precaution_model = build_keras_models(X_train, y_d_train, y_p_train)
    
    # Train models
    print("Training disease classifier...")
    disease_model.fit(
        X_train, y_d_train,
        validation_data=(X_test, y_d_test),
        epochs=10,
        batch_size=32
    )
    
    print("\nTraining precaution predictor...")
    precaution_model.fit(
        X_train, y_p_train,
        validation_data=(X_test, y_p_test),
        epochs=10,
        batch_size=32
    )
    
    return disease_model, precaution_model, vectorizer, disease_encoder, mlb, df

# ============================================
# PREDICTION SYSTEM (Keras version)
# ============================================

class KerasDiseasePredictor:
    def __init__(self, disease_model, precaution_model, vectorizer, disease_encoder, mlb, df):
        self.disease_model = disease_model
        self.precaution_model = precaution_model
        self.vectorizer = vectorizer
        self.disease_encoder = disease_encoder
        self.mlb = mlb
        self.disease_to_precautions = dict(zip(
            df['disease_clean'],
            df['precautions_list']
        ))
    
    def predict(self, symptoms_text):
        """Make predictions using Keras models"""
        try:
            # Vectorize symptoms
            X = self.vectorizer.transform([symptoms_text]).toarray()
            
            # Predict disease
            disease_probs = self.disease_model.predict(X, verbose=0)
            disease_num = np.argmax(disease_probs, axis=1)[0]
            disease = self.disease_encoder.inverse_transform([disease_num])[0]
            
            # Predict precautions
            precautions_bin = self.precaution_model.predict(X, verbose=0) > 0.5
            precautions = self.mlb.inverse_transform(precautions_bin)[0]
            
            # Fallback
            if not precautions and disease in self.disease_to_precautions:
                precautions = self.disease_to_precautions[disease]
            
            return disease, list(precautions) if precautions else []
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None, []

# ============================================
# MODEL PERSISTENCE (.h5 files)
# ============================================

def save_keras_model(predictor, base_filename='disease_predictor'):
    """Save all components of the Keras predictor"""
    try:
        # Save Keras models
        save_model(predictor.disease_model, f'{base_filename}_disease.h5')
        save_model(predictor.precaution_model, f'{base_filename}_precaution.h5')
        
        # Save other components with joblib
        import joblib
        joblib.dump({
            'vectorizer': predictor.vectorizer,
            'disease_encoder': predictor.disease_encoder,
            'mlb': predictor.mlb,
            'disease_to_precautions': predictor.disease_to_precautions
        }, f'{base_filename}_components.joblib')
        
        print(f"Models saved successfully as {base_filename}_*.h5 and components")
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_keras_model(base_filename='disease_predictor'):
    """Load all components of the Keras predictor"""
    try:
        # Load Keras models
        disease_model = load_model(f'{base_filename}_disease.h5')
        precaution_model = load_model(f'{base_filename}_precaution.h5')
        
        # Load other components
        import joblib
        components = joblib.load(f'{base_filename}_components.joblib')
        
        # Recreate predictor
        predictor = KerasDiseasePredictor(
            disease_model=disease_model,
            precaution_model=precaution_model,
            vectorizer=components['vectorizer'],
            disease_encoder=components['disease_encoder'],
            mlb=components['mlb'],
            df=None  # We don't need the original df for prediction
        )
        predictor.disease_to_precautions = components['disease_to_precautions']
        
        print("Keras models loaded successfully")
        return predictor
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# ============================================
# MAIN EXECUTION
# ============================================

def train_and_save_keras_model():
    """Full training and saving workflow with Keras"""
    print("Loading and preprocessing data...")
    df = load_and_preprocess()
    
    if df is None or df.empty:
        print("No valid data available")
        return None
    
    print("\nTraining Keras models...")
    disease_model, precaution_model, vectorizer, disease_encoder, mlb, df = train_keras_models(df)
    
    print("\nCreating predictor...")
    predictor = KerasDiseasePredictor(
        disease_model, precaution_model, vectorizer, disease_encoder, mlb, df)
    
    print("\nTesting predictor...")
    test_cases = [
        "fever chills headache",
        "joint pain stiffness",
        "fatigue weight gain"
    ]
    
    for symptoms in test_cases:
        disease, precautions = predictor.predict(symptoms)
        print(f"\nSymptoms: {symptoms}")
        print(f"Predicted Disease: {disease.capitalize() if disease else 'Unknown'}")
        print("Precautions:", [p.capitalize() for p in precautions] if precautions else "None")
    
    print("\nSaving models...")
    if save_keras_model(predictor):
        return predictor
    return None

if __name__ == "__main__":
    # Train and save the Keras models
    trained_predictor = train_and_save_keras_model()
    
    # Example of loading and using saved models
    if trained_predictor:
        print("\nExample of loading and using saved Keras models:")
        loaded_predictor = load_keras_model()
        if loaded_predictor:
            symptoms = "itching rash swelling"
            disease, precautions = loaded_predictor.predict(symptoms)
            print(f"\nNew Prediction for: {symptoms}")
            print(f"Disease: {disease.capitalize()}")
            print("Precautions:")
            for i, p in enumerate(precautions, 1):
                print(f"{i}. {p.capitalize()}")