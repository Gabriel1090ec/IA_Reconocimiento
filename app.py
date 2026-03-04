import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

st.set_page_config(page_title="Reconocimiento Facial ITSE", layout="centered")

@st.cache_resource
def load_model_and_labels():
    # Cargar modelo (tu MobileNetV2 con 640x480 → 160x160)
    model = tf.keras.models.load_model('mejor_modelo.h5')
    
    # Cargar etiquetas (tu etiquetas.npy)
    if os.path.exists('etiquetas.npy'):
        class_indices = np.load('etiquetas.npy', allow_pickle=True).item()
        class_names = {v: k for k, v in class_indices.items()}
    else:
        class_names = {i: f"Persona_{i}" for i in range(17)}
    
    return model, class_names

model, class_names = load_model_and_labels()

st.title("🎓 Reconocimiento Facial ITSE")
img_file = st.camera_input("Toma una foto")

if img_file:
    # Leer imagen
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # Preprocesamiento IDÉNTICO al entrenamiento:
    # 1. Redimensionar a 640x480
    face_resized = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_AREA)
    
    # 2. Convertir a RGB (3 canales) como hizo tu entrenamiento
    face_rgb = np.stack([face_resized] * 3, axis=-1)
    
    # 3. Normalizar
    face_array = face_rgb.astype('float32') / 255.0
    
    # 4. Añadir dimensión de batch
    face_array = np.expand_dims(face_array, axis=0)
    
    # Predicción
    pred = model.predict(face_array, verbose=0)
    idx = int(np.argmax(pred[0]))
    conf = float(np.max(pred[0])) * 100
    
    # Mostrar resultado
    if conf > 40 and idx in class_names:
        st.success(f"✅ **{class_names[idx]}** ({conf:.1f}%)")
    else:
        st.warning("❓ No reconocido (baja confianza)")
