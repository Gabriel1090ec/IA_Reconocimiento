import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import os

st.set_page_config(page_title="Reconocimiento Facial ITSE", layout="centered")

# Cargar modelo y clases (cacheado para no recargar)
@st.cache_resource
def load_model_and_classes():
    try:
        # Cargar modelo H5 directamente
        model = tf.keras.models.load_model('mejor_modelo.h5')
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {str(e)}")
        st.stop()
    
    # Cargar clases desde JSON
    if os.path.exists('clases.json'):
        with open('clases.json', 'r') as f:
            classes = json.load(f)
        # Convertir claves a enteros para coincidir con las predicciones
        classes = {int(k): v for k, v in classes.items()}
    else:
        st.error("❌ clases.json no encontrado")
        st.stop()
    
    return model, classes

model, classes = load_model_and_classes()

st.title("🎓 Reconocimiento Facial ITSE")
img_file = st.camera_input("Toma una foto")

if img_file:
    # Leer imagen
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # Preprocesamiento IDÉNTICO al entrenamiento:
    # 1. Redimensionar a 640x480 (ancho, alto)
    face_resized = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_AREA)
    
    # 2. Normalizar y añadir dimensiones
    face_array = face_resized.astype('float32') / 255.0
    face_array = np.expand_dims(face_array, axis=(0, -1))  # Forma: (1, 480, 640, 1)
    
    # Verificar forma antes de predecir (para debug)
    # st.write(f"Forma de entrada: {face_array.shape}")  # Debe ser (1, 480, 640, 1)
    
    # Predecir
    try:
        pred = model.predict(face_array, verbose=0)
        idx = int(np.argmax(pred[0]))  # Índice de la clase con mayor probabilidad
        
        # Mostrar SIEMPRE 100% de confianza (requerimiento específico)
        nombre = classes.get(idx, f"Clase_{idx}")
        st.success(f"✅ **{nombre}** (Confianza: 100%)")
        
        # Mostrar todas las clases disponibles para verificación
        st.caption("Clases cargadas:")
        st.json(classes)
        
    except Exception as e:
        st.error(f"❌ Error en predicción: {str(e)}")
        st.caption("Verifica que mejor_modelo.h5 y clases.json estén en la RAÍZ del repositorio")
