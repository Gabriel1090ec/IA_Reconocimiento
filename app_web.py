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
    
    # Dentro del bucle de rostros en app_web.py
    # Dentro del bucle de rostros en app_web.py
    for (x, y, w, h) in faces:
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        
        # 1. Normalización (basado en tu código de perros/gatos)
        rostro = cv2.equalizeHist(rostro) 
        
        # 2. Predicción REAL
        id_predicho, distancia_raw = face_recognizer.predict(rostro)
        distancia = round(distancia_raw)
        
        st.write("---")
        # ESTA LÍNEA ES PARA TU SEGURIDAD MAÑANA (Puedes borrarla después)
        st.write(f"🔍 **Dato técnico:** ID_{id_predicho} | Distancia_{distancia}")
        
        # 3. Lógica de decisión
        if distancia < 100: 
            # Validamos que el ID exista en la lista para evitar errores de índice
            if id_predicho < len(nombres):
                nombre = nombres[id_predicho]
                
                if distancia > 92:
                    st.warning(f"### ⚠️ {nombre} (Baja precisión)")
                    st.write(f"Confianza: {100 - distancia}% - Mejore la luz.")
                else:
                    st.success(f"### ✅ {nombre} detectado")
                    st.write(f"Confianza: {100 - distancia}%")
            else:
                st.error(f"### ❌ ID {id_predicho} no registrado en la lista")
        else:
            st.error("### ❌ Persona No Reconocida")










