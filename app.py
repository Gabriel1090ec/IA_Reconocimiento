import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

# Cargar modelo (cacheado para no recargar en cada interacción)
@st.cache_resource
def load_model_cached():
    return load_model('modelo_cnn.h5')

# Cargar mapeo de clases
@st.cache_resource
def load_classes():
    if os.path.exists('clases.json'):
        with open('clases.json', 'r') as f:
            mapping = json.load(f)
        return [mapping[str(i)] for i in range(len(mapping))]
    return [f"Persona_{i}" for i in range(17)]

model = load_model_cached()
class_names = load_classes()

st.title("🎓 Reconocimiento Facial ITSE")
st.markdown("Sistema de identificación biométrica con Red Neuronal Convolucional")

# Cámara
img = st.camera_input("Toma una foto")

if img:
    # Leer imagen
    bytes_data = img.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostro
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        st.error("❌ No se detectó rostro. Intenta con mejor iluminación.")
    else:
        # Procesar PRIMER rostro detectado
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocesamiento IDÉNTICO al entrenamiento
        face_resized = cv2.resize(face_roi, (640, 480), interpolation=cv2.INTER_AREA)
        face_normalized = cv2.equalizeHist(face_resized)
        face_array = face_normalized.astype('float32') / 255.0
        face_array = np.expand_dims(face_array, axis=(0, -1))  # (1, 480, 640, 1)
        
        # Predicción
        pred = model.predict(face_array, verbose=0)
        idx = np.argmax(pred[0])
        conf = np.max(pred[0]) * 100
        
        # Mostrar resultado
        if conf > 50 and idx < len(class_names):
            st.success(f"✅ **{class_names[idx]}** ({conf:.1f}%)")
        elif conf > 30 and idx < len(class_names):
            st.warning(f"⚠️ **{class_names[idx]}** (baja confianza: {conf:.1f}%)")
        else:
            st.error("❓ No reconocido (confianza muy baja)")
