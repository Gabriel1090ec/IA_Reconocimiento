import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

# Configuración básica
st.set_page_config(
    page_title="Reconocimiento Facial ITSE",
    page_icon="🎓",
    layout="centered"
)

# ══════════════════════════════════════════════════════════════════════════════════════
# CARGAR MODELO Y DATOS
# ══════════════════════════════════════════════════════════════════════════════════════

# Verificar archivos
if not os.path.exists('modelo_entrenado.xml'):
    st.error("❌ Archivo 'modelo_entrenado.xml' no encontrado")
    st.stop()

if not os.path.exists('etiquetas_personas.npy'):
    st.error("❌ Archivo 'etiquetas_personas.npy' no encontrado")
    st.stop()

# Cargar modelo
reconocedor = cv2.face.LBPHFaceRecognizer_create()
reconocedor.read('modelo_entrenado.xml')

# Cargar mapeo de nombres
mapeo_etiquetas = np.load('etiquetas_personas.npy', allow_pickle=True).item()

# Cargar detector de rostros
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ══════════════════════════════════════════════════════════════════════════════════════
# SIDEBAR - SIMPLE Y FUNCIONAL
# ══════════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    try:
        st.image("logo_itse.png", width=250)
    except:
        st.write("ITSE")
    
    st.title("Panel de Control")
    
    st.subheader("ℹ️ Sistema")
    st.write("• Algoritmo: LBPH")
    st.write("• Estudiantes: 10")
    st.write("• Resolución: 150×150 px")
    
    st.subheader("👥 Desarrolladores")
    st.write("1. Gabriel Rodriguez")
    st.write("2. Idney Ayala")
    st.write("3. Josue Fajardo")
    st.write("4. Miguel Herrera")
    st.write("5. Kevin Gonzales")
    
    st.subheader("📋 Registrados")
    for nombre in sorted(mapeo_etiquetas.values()):
        st.write(f"• {nombre}")

# ══════════════════════════════════════════════════════════════════════════════════════
# CUERPO PRINCIPAL - LIMPIO Y LEGIBLE
# ══════════════════════════════════════════════════════════════════════════════════════

st.title("🎓 Reconocimiento Facial ITSE")
st.markdown("Sistema de identificación biométrica para estudiantes del grupo")

st.info("""
**📸 Instrucciones:**
1. Haz clic en el botón de cámara abajo
2. Permite el acceso a tu cámara web
3. Enfoca tu rostro directamente con buena iluminación frontal
""")

# Captura de imagen
img_file = st.camera_input("Tomar foto")

if img_file:
    with st.spinner("Analizando rostro..."):
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
        
        # Predecir
        id_predicho, distancia = reconocedor.predict(rostro)
        confianza = max(0, 100 - distancia)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Mostrar resultado con componentes nativos de Streamlit
        st.divider()
        
        if id_predicho in mapeo_etiquetas and distancia < 100:
            nombre = mapeo_etiquetas[id_predicho]
            
            if confianza >= 70:
                st.success(f"✅ {nombre} detectado")
                st.write(f"**Confianza:** {confianza:.0f}%")
                st.write(f"**Hora:** {timestamp}")
            
            elif confianza >= 50:
                st.warning(f"⚠️ {nombre} (baja precisión)")
                st.write(f"**Confianza:** {confianza:.0f}%")
                st.write("💡 Recomendación: Mejora la iluminación y vuelve a intentar")
            
            else:
                st.warning(f"❓ {nombre} (confianza muy baja)")
                st.write(f"**Confianza:** {confianza:.0f}%")
                st.write("⚠️ La confianza es insuficiente para una identificación fiable")
        else:
            st.error("""
            ❌ Persona no reconocida
            
            El rostro no coincide con estudiantes registrados en el sistema.
            """)
        
        st.divider()

# Pie de página
st.caption("Instituto Tecnológico Superior Especializado (ITSE) • Proyecto de Visión Artificial")


