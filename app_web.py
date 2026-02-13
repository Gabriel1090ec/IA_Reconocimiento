import streamlit as st
import cv2
import numpy as np
import os

# Configuración de la página
st.set_page_config(
    page_title="IA Reconocimiento ITSE",
    page_icon="🤖",
    layout="centered"
)

# Estilos CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════════
# CARGAR ARCHIVOS GENERADOS POR EL ENTRENADOR
# ══════════════════════════════════════════════════════════════════════════════════════

# Verificar existencia de archivos
if not os.path.exists('modelo_entrenado.xml'):
    st.error("❌ Archivo 'modelo_entrenado.xml' no encontrado")
    st.stop()

if not os.path.exists('etiquetas_personas.npy'):
    st.error("❌ Archivo 'etiquetas_personas.npy' no encontrado")
    st.stop()

# Cargar modelo entrenado
reconocedor = cv2.face.LBPHFaceRecognizer_create()
reconocedor.read('modelo_entrenado.xml')

# Cargar mapeo de etiquetas (ID -> Nombre)
mapeo_etiquetas = np.load('etiquetas_personas.npy', allow_pickle=True).item()

# Cargar clasificador Haar Cascade para detección de rostros
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ══════════════════════════════════════════════════════════════════════════════════════
# BARRA LATERAL
# ══════════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://www.itse.ac.pa/logo.png", width=150)
    st.title("Panel de Control")
    st.info("Sistema de reconocimiento facial basado en algoritmo LBPH.")
    st.write("---")
    st.write("**Equipo de Desarrollo:**")
    st.write("1. Gabriel Rodriguez")
    st.write("2. Idney Ayala")
    st.write("3. Josue Fajardo")
    st.write("4. Miguel Herrera")
    st.write("5. Kevin Gonzales")
    
    st.write("---")
    st.write("**Estudiantes Registrados:**")
    for nombre in sorted(mapeo_etiquetas.values()):
        st.write(f"• {nombre}")

# ══════════════════════════════════════════════════════════════════════════════════════
# CUERPO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════════════

st.title("🤖 Sistema de Reconocimiento Facial")
st.subheader("Identificación de estudiantes del grupo ITSE")

# Captura de imagen
img_file = st.camera_input("Enfoca tu rostro frente a la cámara")

if img_file:
    with st.spinner("Analizando imagen..."):
        # Convertir a formato OpenCV
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        # Detección de rostros
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            st.warning("⚠️ No se detectó ningún rostro. Asegúrate de estar bien iluminado y mirar a la cámara.")
            st.stop()
        
        # Procesar el rostro más grande
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]
        
        # Extraer y preprocesar rostro
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_AREA)
        rostro = cv2.equalizeHist(rostro)
        
        # Predicción usando el modelo entrenado
        id_predicho, distancia = reconocedor.predict(rostro)
        confianza = max(0, 100 - distancia)
        
        # Mostrar resultado
        st.write("---")
        
        if id_predicho in mapeo_etiquetas and distancia < 100:
            nombre = mapeo_etiquetas[id_predicho]
            
            if confianza >= 70:
                st.success(f"✅ {nombre} detectado")
                st.write(f"Confianza: {confianza:.0f}%")
            elif confianza >= 50:
                st.warning(f"⚠️ {nombre} (Baja precisión)")
                st.write(f"Confianza: {confianza:.0f}% - Mejore la iluminación")
            else:
                st.warning(f"❓ {nombre} (Confianza muy baja)")
                st.write(f"Confianza: {confianza:.0f}%")
        else:
            st.error("❌ Persona no reconocida")

# ══════════════════════════════════════════════════════════════════════════════════════
# PIE DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption("Sistema desarrollado para ITSE • Algoritmo: LBPH Face Recognizer")
