import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from tensorflow.keras.models import load_model
import pickle

# Configuración básica
st.set_page_config(
    page_title="Reconocimiento Facial ITSE - CNN",
    page_icon="🎓",
    layout="centered"
)

# ══════════════════════════════════════════════════════════════════════════════════════
# CARGAR MODELO CNN Y DATOS
# ══════════════════════════════════════════════════════════════════════════════════════

if not os.path.exists('modelo_cnn.h5'):
    st.error("❌ Archivo 'modelo_cnn.h5' no encontrado")
    st.stop()

if not os.path.exists('label_encoder.pkl'):
    st.error("❌ Archivo 'label_encoder.pkl' no encontrado")
    st.stop()

# Cargar modelo CNN
try:
    model = load_model('modelo_cnn.h5')
    st.sidebar.success("✅ Modelo CNN cargado")
except Exception as e:
    st.error(f"❌ Error al cargar modelo: {str(e)}")
    st.stop()

# Cargar label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Cargar detector de rostros
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ══════════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    if os.path.exists("logo_itse.png"):
        st.image("logo_itse.png", width=150)
    else:
        st.markdown("### 🎓 ITSE")
    
    st.title("Panel de Control")
    
    st.subheader("ℹ️ Sistema")
    st.write("• Algoritmo: CNN (Red Neuronal)")
    st.write("• Arquitectura: 3 bloques convolucionales")
    st.write(f"• Clases: {len(label_encoder.classes_)}")
    st.write("• Resolución: 150×150 px")
    
    st.subheader("👥 Desarrolladores")
    st.write("1. Gabriel Rodriguez")
    st.write("2. Idney Ayala")
    st.write("3. Josue Fajardo")
    st.write("4. Miguel Herrera")
    st.write("5. Kevin Gonzales")
    
    st.subheader("📋 Registrados")
    for nombre in sorted(label_encoder.classes_):
        st.write(f"• {nombre}")

# ══════════════════════════════════════════════════════════════════════════════════════
# CUERPO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════════════

st.title("🎓 Reconocimiento Facial ITSE - Red Neuronal")
st.markdown("Sistema de identificación biométrica usando Deep Learning")

st.info("""
**📸 Instrucciones:**
1. Haz clic en el botón de cámara abajo
2. Permite el acceso a tu cámara web
3. Enfoca tu rostro directamente con buena iluminación frontal
""")

# Captura de imagen
img_file = st.camera_input("Tomar foto")

if img_file:
    with st.spinner("🧠 Analizando rostro con Red Neuronal..."):
        # Convertir a formato OpenCV
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            st.error("""
            ❌ No se detectó ningún rostro
            
            **Recomendaciones:**
            • Asegúrate de tener buena iluminación frontal
            • Mira directamente a la cámara
            • Mantén el rostro completamente visible
            """)
            st.stop()
        
        # Procesar el rostro más grande
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_AREA)
        rostro = cv2.equalizeHist(rostro)
        
        # Preprocesar para CNN
        rostro_array = np.array(rostro, dtype='float32') / 255.0
        rostro_array = np.expand_dims(rostro_array, axis=(0, -1))  # (1, 150, 150, 1)
        
        # Predecir con CNN
        predictions = model.predict(rostro_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100
        nombre = label_encoder.classes_[class_idx]
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Mostrar resultado
        st.divider()
        
        if confidence >= 70:
            st.success(f"✅ {nombre} detectado")
            st.write(f"**Confianza:** {confidence:.1f}%")
            st.write(f"**Hora:** {timestamp}")
        elif confidence >= 50:
            st.warning(f"⚠️ {nombre} (baja precisión)")
            st.write(f"**Confianza:** {confidence:.1f}%")
            st.write("💡 Recomendación: Mejora la iluminación y vuelve a intentar")
        else:
            st.error("""
            ❌ Persona no reconocida
            
            El rostro no coincide con estudiantes registrados en el sistema.
            """)
        
        st.divider()

# Pie de página
st.caption("Instituto Tecnológico Superior Especializado (ITSE) • Proyecto de Deep Learning")