import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import os

st.set_page_config(page_title="Reconocimiento Facial", layout="centered")

@st.cache_resource
def load_model_and_classes():
    try:
        model = tf.keras.models.load_model('mejor_modelo.h5')
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {str(e)}")
        st.stop()
    
    if os.path.exists('clases.json'):
        with open('clases.json', 'r') as f:
            classes = json.load(f)
    else:
        classes = {str(i): f"Persona_{i}" for i in range(17)}
    
    return model, classes

model, classes = load_model_and_classes()

st.title("🎓 Reconocimiento Facial ITSE")
img_file = st.camera_input("Toma una foto")

if img_file:
    # Leer imagen
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Convertir a escala de grises (1 canal)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostro
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        st.error("❌ No se detectó rostro")
    else:
        # Recortar rostro
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # REDIMENSIONAR CORRECTAMENTE: (ancho=640, alto=480)
        # cv2.resize usa (ancho, alto) → resultado NumPy: (480, 640)
        face_resized = cv2.resize(face_roi, (640, 480), interpolation=cv2.INTER_AREA)
        
        # Normalizar y añadir dimensiones
        # Forma final: (1, 480, 640, 1) ← EXACTAMENTE lo que espera el modelo
        face_array = face_resized.astype('float32') / 255.0
        face_array = np.expand_dims(face_array, axis=0)  # Añadir batch dimension → (1, 480, 640)
        face_array = np.expand_dims(face_array, axis=-1)  # Añadir canal → (1, 480, 640, 1)
        
        # Verificar forma antes de predecir (para debug)
        # st.write(f"Forma de entrada: {face_array.shape}")  # Debe ser (1, 480, 640, 1)
        
        # Predecir
        pred = model.predict(face_array, verbose=0)
        idx = int(np.argmax(pred[0]))
        conf = float(np.max(pred[0])) * 100
        
        # Mostrar resultado
        if conf > 40 and str(idx) in classes:
            st.success(f"✅ **{classes[str(idx)]}** ({conf:.1f}%)")
        else:
            st.warning("❓ No reconocido (baja confianza)")
