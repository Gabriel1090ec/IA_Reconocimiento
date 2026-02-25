import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime


st.set_page_config(
    page_title="Reconocimiento Facial ITSE",
    page_icon="🎓",
    layout="centered"
)

# ══════════════════════════════════════════════════════════════════════════════════════
# CARGAR MODELO LBPH
# ══════════════════════════════════════════════════════════════════════════════════════

if not os.path.exists('modelo_lbph.xml'):
    st.error("❌ Modelo no encontrado: modelo_lbph.xml")
    st.stop()

if not os.path.exists('etiquetas_personas.npy'):
    st.error("❌ Etiquetas no encontradas: etiquetas_personas.npy")
    st.stop()

# Cargar modelo LBPH
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
        threshold=100  # ← UMORAL PERMISIVO
    )
    recognizer.read('modelo_lbph.xml')
    st.sidebar.success("✅ Modelo cargado")
except Exception as e:
    st.error(f"❌ Error al cargar modelo: {str(e)}")
    st.stop()

# Cargar etiquetas
label_map = np.load('etiquetas_personas.npy', allow_pickle=True).item()

# Cargar detector
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ══════════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🎓 ITSE")
    st.title("Panel de Control")
    
    st.subheader("ℹ️ Sistema")
    st.write("• Algoritmo: Haar Cascade para detección de rostros")
    st.write("• Umbral: 100")
    st.write(f"• Personas: {len(label_map)}")
    st.write("• Resolución: 150×150 px")
    
    st.subheader("👥 Desarrolladores")
    st.write("1. Gabriel Rodriguez")
    st.write("2. Idney Ayala")
    st.write("3. Josue Fajardo")
    st.write("4. Miguel Herrera")
    st.write("5. Kevin Gonzales")
    
    st.subheader("📋 Registrados")
    for nombre in sorted(label_map.values()):
        st.write(f"• {nombre}")

# ══════════════════════════════════════════════════════════════════════════════════════
# CUERPO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════════════

st.title("🎓 Reconocimiento Facial ITSE ")
st.markdown("Sistema de identificación biométrica")

st.info("""
**📸 Instrucciones:**
1. Haz clic en el botón de cámara
2. Permite acceso a tu cámara web
3. Enfoca tu rostro con buena iluminación frontal
""")

# Captura
img_file = st.camera_input("Tomar foto")

if img_file:
    with st.spinner("🔍 Analizando rostro..."):
        # Convertir a OpenCV
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            st.error("❌ No se detectó ningún rostro")
            st.stop()
        
        # Procesar rostro más grande
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]
        rostro = gray[y:y+h, x:x+w]
        
        # 🔑 PREPROCESAMIENTO IGUAL AL ENTRENAMIENTO
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_AREA)
        rostro = cv2.equalizeHist(rostro)
        
        # Predecir
        id_predicho, distancia = recognizer.predict(rostro)
        confianza = max(0, 100 - distancia)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Mostrar resultado
        st.divider()
        
        # 🔑 UMBRAL CORREGIDO: distancia < 100 (más permisivo)
        if id_predicho in label_map and distancia < 100:
            nombre = label_map[id_predicho]
            
            if confianza >= 70:
                st.success(f"✅ {nombre} reconocido")
                st.write(f"**Confianza:** {confianza:.0f}%")
                st.write(f"**Distancia:** {distancia:.1f}")
                st.write(f"**Hora:** {timestamp}")
            
            elif confianza >= 40:
                st.warning(f"⚠️ {nombre} (baja confianza)")
                st.write(f"**Confianza:** {confianza:.0f}%")
                st.write(f"**Distancia:** {distancia:.1f}")
            
            else:
                st.info(f"❓ Posible {nombre} (confianza muy baja)")
                st.write(f"**Confianza:** {confianza:.0f}%")
                st.write(f"**Distancia:** {distancia:.1f}")
        else:
            st.error("❌ No reconocido")
            st.write(f"**Distancia:** {distancia:.1f}")
            st.write("El rostro no coincide con personas registradas")
        
        st.divider()


st.caption("ITSE • Proyecto de Visión Artificial")

