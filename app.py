import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import json

st.set_page_config(page_title="Reconocimiento Facial", layout="centered")

@st.cache_resource
def load_model_and_data():
    try:
        model = tf.keras.models.load_model('modelo_cnn.h5')
    except:
        st.error("❌ Error: modelo_cnn.h5 no encontrado")
        st.stop()
    
    if os.path.exists('clases.json'):
        with open('clases.json', 'r') as f:
            class_map = json.load(f)
        class_names = [class_map[str(i)] for i in range(len(class_map))]
    else:
        class_names = [f"Persona_{i+1}" for i in range(17)]
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    return model, class_names, face_cascade

model, class_names, face_cascade = load_model_and_data()

st.title("🎓 Reconocimiento Facial")
st.markdown("Sistema de identificación biométrica con Red Neuronal Convolucional")

img_file = st.camera_input("Toma una foto")

if img_file:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        st.error("❌ No se detectó rostro")
    else:
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        face_resized = cv2.resize(face_roi, (640, 480), interpolation=cv2.INTER_AREA)
        face_normalized = cv2.equalizeHist(face_resized)
        face_array = face_normalized.astype('float32') / 255.0
        face_array = np.expand_dims(face_array, axis=(0, -1))
        
        pred = model.predict(face_array, verbose=0)
        idx = int(np.argmax(pred[0]))
        conf = float(np.max(pred[0])) * 100
        
        if conf > 40 and idx < len(class_names):
            st.success(f"✅ **{class_names[idx]}** ({conf:.1f}%)")
        else:
            st.warning("❓ No reconocido (baja confianza)")
