import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LA PÁGINA
# ══════════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="IA Reconocimiento ITSE",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════════════
# ESTILOS CSS PROFESIONALES
# ══════════════════════════════════════════════════════════════════════════════════════

st.markdown("""
    <style>
    /* Fuente principal */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Fondo principal */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Contenido principal con fondo blanco */
    [data-testid="stMainBlockContainer"] {
        background: transparent;
    }
    
    .main-content {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        max-width: 800px;
        margin: 2rem auto;
    }
    
    /* Header */
    .header-section {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .header-section h1 {
        color: #2d3748;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .header-section p {
        color: #718096;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Cards de resultado */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }
    
    .result-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .result-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .result-error {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .result-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .result-card p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    
    /* Botones */
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: white !important;
        box-shadow: 2px 0 15px rgba(0, 0, 0, 0.08);
    }
    
    [data-testid="stSidebar"] .css-1v0mbdj {
        margin-top: 1rem;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #2d3748 !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }
    
    [data-testid="stSidebar"] .stInfo {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
    }
    
    .sidebar-section {
        margin: 1.5rem 0;
    }
    
    .sidebar-section h4 {
        color: #4a5568;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .sidebar-section ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .sidebar-section li {
        padding: 0.4rem 0;
        color: #4a5568;
        font-size: 0.95rem;
    }
    
    .sidebar-section li:before {
        content: "• ";
        color: #3b82f6;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #e2e8f0, transparent);
        margin: 2rem 0;
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
# BARRA LATERAL - DISEÑO PROFESIONAL
# ══════════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://www.itse.ac.pa/logo.png", width=180)
    st.markdown("### 🎓 ITSE")
    st.markdown("**Sistema de Reconocimiento Facial**")
    
    st.info("🔍 Algoritmo: LBPH Face Recognizer\n\n📊 Personas registradas: 10")
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 👥 Equipo de Desarrollo")
    st.markdown("""
    <ul>
        <li>Gabriel Rodriguez</li>
        <li>Idney Ayala</li>
        <li>Josue Fajardo</li>
        <li>Miguel Herrera</li>
        <li>Kevin Gonzales</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 📋 Estudiantes Registrados")
    for nombre in sorted(mapeo_etiquetas.values()):
        st.markdown(f"<li>{nombre}</li>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### ℹ️ Información")
    st.markdown("""
    <ul>
        <li>Modelo: LBPH</li>
        <li>Umbral: 100</li>
        <li>Resolución: 150×150 px</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════════
# CUERPO PRINCIPAL - DISEÑO MODERNO
# ══════════════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-section">
        <h1>🤖 Reconocimiento Facial ITSE</h1>
        <p>Sistema de identificación biométrica para estudiantes</p>
    </div>
""", unsafe_allow_html=True)

# Descripción
st.markdown("""
<div style="background: #f8fafc; padding: 1.2rem; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 1.5rem;">
    <p style="margin: 0; color: #4a5568; line-height: 1.6;">
        <strong>📸 Instrucciones:</strong> Haz clic en el botón de cámara, permite el acceso y enfoca tu rostro 
        directamente a la cámara con buena iluminación. El sistema identificará automáticamente a los estudiantes registrados.
    </p>
</div>
""", unsafe_allow_html=True)

# Captura de imagen
img_file = st.camera_input("📷 Capturar rostro")

if img_file:
    with st.spinner("🔄 Analizando imagen..."):
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
            st.markdown("""
                <div class="result-card result-error">
                    <h3>❌ No se detectó rostro</h3>
                    <p>Asegúrate de: estar bien iluminado, mirar directamente a la cámara, 
                    y tener el rostro completamente visible sin obstrucciones.</p>
                </div>
            """, unsafe_allow_html=True)
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
        
        # Mostrar resultado con diseño profesional
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        if id_predicho in mapeo_etiquetas and distancia < 100:
            nombre = mapeo_etiquetas[id_predicho]
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if confianza >= 70:
                st.markdown(f"""
                    <div class="result-card result-success">
                        <h3>✅ {nombre}</h3>
                        <p><strong>Estado:</strong> Identificación exitosa</p>
                        <p><strong>Confianza:</strong> {confianza:.0f}%</p>
                        <p><strong>Hora:</strong> {timestamp}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            elif confianza >= 50:
                st.markdown(f"""
                    <div class="result-card result-warning">
                        <h3>⚠️ {nombre}</h3>
                        <p><strong>Estado:</strong> Identificación con baja precisión</p>
                        <p><strong>Confianza:</strong> {confianza:.0f}%</p>
                        <p><strong>Recomendación:</strong> Mejora la iluminación y vuelve a intentar</p>
                    </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                    <div class="result-card result-warning">
                        <h3>❓ {nombre}</h3>
                        <p><strong>Estado:</strong> Identificación incierta</p>
                        <p><strong>Confianza:</strong> {confianza:.0f}%</p>
                        <p><strong>Nota:</strong> La confianza es muy baja para una identificación fiable</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-card result-error">
                    <h3>❌ Persona no reconocida</h3>
                    <p><strong>Estado:</strong> No coincide con estudiantes registrados</p>
                    <p><strong>Posibles causas:</strong> Rostro no registrado, mala iluminación, 
                    o ángulo inadecuado de captura.</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════════
# PIE DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════════════

st.markdown("""
    <div class="footer">
        <p>🎓 Instituto Tecnológico Superior Especializado (ITSE) • Proyecto de Visión Artificial</p>
        <p>🤖 Algoritmo: LBPH Face Recognizer • Versión 1.0</p>
    </div>
""", unsafe_allow_html=True)
