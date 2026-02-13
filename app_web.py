import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="IA Reconocimiento ITSE",
    page_icon="🎓",
    layout="centered"
)

# CSS minimalista y 100% legible - SIN colores que tapen texto
st.markdown("""
    <style>
    /* Fondo claro suave */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* Contenedor principal blanco con borde sutil */
    .main-container {
        background-color: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        max-width: 800px;
        margin: 2rem auto;
    }
    
    /* Títulos oscuros */
    .main-container h1 {
        color: #1e293b;
        font-weight: 700;
    }
    
    .main-container h2 {
        color: #334155;
        font-weight: 600;
    }
    
    .main-container h3 {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Texto principal oscuro */
    .main-container p, .main-container li {
        color: #334155;
        line-height: 1.6;
    }
    
    /* Cards de resultado - fondos claros con texto oscuro */
    .card-success {
        background-color: #ecfdf5;
        border-left: 4px solid #10b981;
        color: #065f46;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.2rem 0;
    }
    
    .card-warning {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        color: #92400e;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.2rem 0;
    }
    
    .card-error {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        color: #b91c1c;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.2rem 0;
    }
    
    /* Sidebar blanco con texto oscuro */
    [data-testid="stSidebar"] {
        background-color: white !important;
    }
    
    [data-testid="stSidebar"] .css-1v0mbdj img {
        margin-top: 1rem;
    }
    
    /* Divider sutil */
    .divider {
        height: 1px;
        background-color: #e2e8f0;
        margin: 1.5rem 0;
    }
    
    /* Botón con contraste alto */
    .stButton button {
        background-color: #1e40af;
        color: white;
        border: none;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    
    .stButton button:hover {
        background-color: #1d4ed8;
    }
    
    /* Footer */
    .footer {
        color: #64748b;
        text-align: center;
        padding: 1rem;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

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
# SIDEBAR - SIMPLE Y LEGIBLE
# ══════════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://www.itse.ac.pa/logo.png", width=150)
    st.title("ITSE")
    st.markdown("**Sistema de Reconocimiento Facial**")
    
    st.markdown("### ℹ️ Información")
    st.markdown("- Algoritmo: LBPH")
    st.markdown("- Estudiantes: 10")
    st.markdown("- Resolución: 150×150 px")
    
    st.markdown("### 👥 Desarrolladores")
    st.markdown("1. Gabriel Rodriguez")
    st.markdown("2. Idney Ayala")
    st.markdown("3. Josue Fajardo")
    st.markdown("4. Miguel Herrera")
    st.markdown("5. Kevin Gonzales")
    
    st.markdown("### 📋 Registrados")
    for nombre in sorted(mapeo_etiquetas.values()):
        st.markdown(f"- {nombre}")

# ══════════════════════════════════════════════════════════════════════════════════════
# CONTENEDOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Título
st.title("🎓 Reconocimiento Facial ITSE")
st.subheader("Sistema de identificación biométrica para estudiantes")

# Instrucciones
st.info("""
📸 **Instrucciones:** 
1. Haz clic en "Tomar foto" 
2. Permite el acceso a la cámara
3. Enfoca tu rostro directamente con buena iluminación frontal
""")

# Captura
img_file = st.camera_input("Tomar foto")

if img_file:
    with st.spinner("Analizando rostro..."):
        # Procesar imagen
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            st.markdown("""
                <div class="card-error">
                    <h3>❌ No se detectó rostro</h3>
                    <p>• Asegúrate de tener buena iluminación frontal</p>
                    <p>• Mira directamente a la cámara</p>
                    <p>• Mantén el rostro visible sin obstrucciones</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div><div class="footer">ITSE • Proyecto de Visión Artificial</div>', unsafe_allow_html=True)
            st.stop()
        
        # Procesar rostro más grande
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_AREA)
        rostro = cv2.equalizeHist(rostro)
        
        # Predecir
        id_predicho, distancia = reconocedor.predict(rostro)
        confianza = max(0, 100 - distancia)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Mostrar resultado
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        if id_predicho in mapeo_etiquetas and distancia < 100:
            nombre = mapeo_etiquetas[id_predicho]
            
            if confianza >= 70:
                st.markdown(f"""
                    <div class="card-success">
                        <h3>✅ {nombre}</h3>
                        <p><strong>Estado:</strong> Identificación exitosa</p>
                        <p><strong>Confianza:</strong> {confianza:.0f}%</p>
                        <p><strong>Hora:</strong> {timestamp}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            elif confianza >= 50:
                st.markdown(f"""
                    <div class="card-warning">
                        <h3>⚠️ {nombre}</h3>
                        <p><strong>Estado:</strong> Baja precisión</p>
                        <p><strong>Confianza:</strong> {confianza:.0f}%</p>
                        <p><strong>Recomendación:</strong> Mejora la iluminación</p>
                    </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown(f"""
                    <div class="card-warning">
                        <h3>❓ {nombre}</h3>
                        <p><strong>Confianza:</strong> {confianza:.0f}% (muy baja)</p>
                        <p><strong>Nota:</strong> No es suficiente para identificación fiable</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="card-error">
                    <h3>❌ Persona no reconocida</h3>
                    <p>El rostro no coincide con estudiantes registrados en el sistema.</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown('</div><div class="footer">Instituto Tecnológico Superior Especializado (ITSE) • Proyecto de Visión Artificial</div>', unsafe_allow_html=True)
