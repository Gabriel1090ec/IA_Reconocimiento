import streamlit as st
import cv2
import numpy as np

# 1. Configuración de la Estética de la Página
st.set_page_config(
    page_title="IA Reconocimiento ITSE",
    page_icon="🤖",
    layout="centered"
)

# Estilo personalizado con Markdown (opcional para colores)
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Barra Lateral (Sidebar)
with st.sidebar:
    st.image("https://www.itse.ac.pa/logo.png", width=150) # Pon un logo si tienes
    st.title("Panel de Control")
    st.info("Este sistema utiliza el algoritmo LBPH para reconocer a los estudiantes del grupo.")
    st.write("---")
    st.write("**Integrantes:**")
    st.write("1- Gabriel Rodriguez (Desarrollador)")
    st.write("2. Idney Ayala (Desarrollador)")
    st.write("3. Josue Fajardo (Desarrollador)")
    st.write("4. Miguel Herrera (Desarrollador)")
    st.write("4. Kevin Gonzales (Desarrollador)")

# 3. Cuerpo Principal
st.title("🤖 Sistema de Visión Artificial")
st.subheader("Reconocimiento Facial en Tiempo Real")

# Cargar Modelo y Cascade
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modelo_entrenado.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
nombres = ['Daniela', 'Elohim', 'Gabriel', 'Idney', 'Kevin', 'Miguel', 'Patricia', 'Roberto', 'Victor']

# Contenedor para la cámara
with st.container():
    img_file = st.camera_input("Enfoca tu rostro frente a la cámara")

if img_file:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        st.warning("No se detectó ningún rostro. Intenta acercarte más.")
    
    for (x, y, w, h) in faces:
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        
        id_detectado = result[0]
        distancia = round(result[1])
        
        st.write("---")
        
        # UMBRAL EQUILIBRADO: 105 es generoso para el celular
        if distancia < 105: 
            nombre = nombres[result[0]]
            
            # Si la distancia es muy alta (entre 92 y 105), es una zona de duda
            if distancia > 20:
                st.warning(f"### ⚠️ {nombre} (Baja precisión)")
                st.write(f"Confianza: {100 - distancia}% - Mejore la luz para confirmar.")
            else:
                st.success(f"### ✅ {nombre} detectado")
                st.write(f"Confianza: {100 - distancia}%")
        else:
            st.error("### ❌ Persona No Reconocida")








