import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento Facial - Redes Neuronales",
    page_icon="🎓",
    layout="centered"
)

# CSS personalizado para mejorar la visualización
st.markdown("""
    <style>
    .confidence-high { color: #00C851; font-weight: bold; font-size: 24px; }
    .confidence-medium { color: #ffbb33; font-weight: bold; font-size: 24px; }
    .confidence-low { color: #ff4444; font-weight: bold; font-size: 24px; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Carga el modelo y las etiquetas una sola vez"""
    try:
        model = tf.keras.models.load_model('mejor_modelo.h5')
        class_dict = np.load('etiquetas.npy', allow_pickle=True).item()
        labels = {v: k for k, v in class_dict.items()}
        
        # Cargar clasificador de rostros de OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        return model, labels, face_cascade
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None, None

def detect_and_crop_face(image, face_cascade):
    """
    Detecta el rostro y lo recorta para mejorar la precisión.
    Si no detecta rostro, usa la imagen completa.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Tomar el rostro más grande (probablemente el más cercano)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Agregar margen del 20%
        margin = int(0.2 * w)
        x, y = max(0, x-margin), max(0, y-margin)
        w, h = w + 2*margin, h + 2*margin
        face_crop = gray[y:y+h, x:x+w]
        return face_crop, True, (x, y, w, h)
    else:
        # Si no hay rostro, convertir a gray y usar completa
        return gray, False, None

def preprocess_image(image, face_cascade):
    """Preprocesa la imagen con mejora de contraste"""
    # Detectar y recortar rostro
    face_img, face_detected, coords = detect_and_crop_face(image, face_cascade)
    
    # Redimensionar a 640x480
    face_img = cv2.resize(face_img, (640, 480))
    
    # Mejora de contraste (Ecualización de histograma adaptativa)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face_img = clahe.apply(face_img.astype(np.uint8))
    
    # Normalizar
    face_img = face_img / 255.0
    
    # Agregar dimensiones: (1, 480, 640, 1)
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    
    return face_img, face_detected, coords

def get_confidence_color(confidence):
    """Retorna color según nivel de confianza"""
    if confidence >= 80:
        return "confidence-high", "🟢"
    elif confidence >= 50:
        return "confidence-medium", "🟡"
    else:
        return "confidence-low", "🔴"

def main():
    st.title("🎓 Reconocimiento Facial - Compañeros")
    st.write("Sistema de reconocimiento usando Redes Neuronales (TensorFlow) con detección de rostro")
    
    # Cargar modelo
    model, labels, face_cascade = load_model()
    
    if model is None:
        st.error("No se pudo cargar el modelo. Verifica que 'mejor_modelo.h5' y 'etiquetas.npy' existan.")
        return
    
    st.success(f"✅ Modelo cargado: **{len(labels)}** compañeros detectados")
    
    # Sidebar con configuraciones
    with st.sidebar:
        st.header("⚙️ Configuración")
        min_confidence = st.slider("Umbral mínimo de confianza", 0, 100, 50)
        show_all_predictions = st.checkbox("Mostrar todas las predicciones (Top 5)", True)
        strict_mode = st.checkbox("Modo estricto (solo mostrar si >80% confianza)", False)
        st.info("💡 **Consejo:** Asegúrate de que tu rostro esté bien iluminado y centrado para mejor precisión.")
    
    # Opciones de entrada
    option = st.radio("Selecciona método de entrada:", 
                      ["📷 Usar Cámara", "📁 Subir Imagen"], 
                      horizontal=True)
    
    image_input = None
    
    if option == "📷 Usar Cámara":
        camera_image = st.camera_input("Toma una foto")
        if camera_image is not None:
            image_input = Image.open(camera_image)
    else:
        uploaded_file = st.file_uploader("Elige una imagen...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image_input = Image.open(uploaded_file)
    
    if image_input is not None:
        img_array = np.array(image_input)
        
        # Mostrar imagen original
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_input, caption="Imagen Original", use_column_width=True)
        
        # Botón para analizar
        if st.button("🔍 Identificar Persona", type="primary"):
            with st.spinner("Detectando rostro y analizando..."):
                try:
                    # Preprocesar (con detección de rostro)
                    processed, face_detected, coords = preprocess_image(img_array, face_cascade)
                    
                    # Mostrar si se detectó rostro
                    if face_detected:
                        st.success("✅ Rostro detectado y recortado automáticamente")
                        # Dibujar rectángulo en la imagen
                        img_with_box = img_array.copy()
                        x, y, w, h = coords
                        cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        with col2:
                            st.image(img_with_box, caption="Detección de Rostro", use_column_width=True)
                    else:
                        st.warning("⚠️ No se detectó un rostro claro. Se usará la imagen completa (puede afectar la precisión).")
                        with col2:
                            st.image(image_input, caption="Sin detección de rostro", use_column_width=True)
                    
                    # PREDICCIÓN
                    predictions = model.predict(processed, verbose=0)[0]
                    
                    # Obtener top 5
                    top_indices = np.argsort(predictions)[::-1][:5]
                    top_predictions = [(labels.get(i, "Desconocido"), float(predictions[i]) * 100) 
                                      for i in top_indices]
                    
                    best_class = top_indices[0]
                    best_confidence = top_predictions[0][1]
                    person_name = top_predictions[0][0]
                    
                    # Verificar umbral mínimo
                    if best_confidence < min_confidence:
                        st.error(f"❌ Confianza muy baja ({best_confidence:.1f}%). Intenta con mejor iluminación o acércate más a la cámara.")
                        return
                    
                    if strict_mode and best_confidence < 80:
                        st.warning("🔒 Modo estricto activado: Se requiere >80% de confianza para mostrar resultados.")
                        return
                    
                    # Mostrar resultado principal
                    st.divider()
                    st.subheader("🎯 Resultado Principal")
                    
                    color_class, emoji = get_confidence_color(best_confidence)
                    
                    # Tarjeta de resultado
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{emoji} Persona Identificada</h3>
                        <h2 style="margin:0;">{person_name}</h2>
                        <p class="{color_class}">Confianza: {best_confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Barra de progreso visual
                    st.progress(int(best_confidence))
                    
                    # Mensaje según confianza
                    if best_confidence >= 90:
                        st.balloons()
                        st.success("🎉 **¡Identificación exitosa con alta confianza!**")
                    elif best_confidence >= 70:
                        st.success("✅ Identificación correcta")
                    elif best_confidence >= 50:
                        st.warning("⚠️ Identificación incierta - Podría ser otra persona similar")
                    else:
                        st.error("❌ Baja confianza - No se puede determinar con certeza")
                    
                    # Mostrar todas las predicciones si está activado
                    if show_all_predictions:
                        st.divider()
                        st.subheader("📊 Top 5 Predicciones")
                        
                        # Crear DataFrame para gráfico
                        df = pd.DataFrame(top_predictions, columns=['Persona', 'Confianza'])
                        df = df.sort_values('Confianza', ascending=True)
                        
                        # Gráfico de barras
                        st.bar_chart(df.set_index('Persona'))
                        
                        # Tabla detallada
                        st.dataframe(
                            df.sort_values('Confianza', ascending=False),
                            column_config={
                                "Confianza": st.column_config.ProgressColumn(
                                    "Confianza %",
                                    help="Nivel de confianza de la predicción",
                                    format="%.1f%%",
                                    min_value=0,
                                    max_value=100,
                                )
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    # Información técnica
                    with st.expander("🔧 Ver información técnica"):
                        st.write(f"**Forma del tensor de entrada:** {processed.shape}")
                        st.write(f"**Rango de valores:** [{processed.min():.3f}, {processed.max():.3f}]")
                        st.write(f"**Rostro detectado:** {'Sí' if face_detected else 'No'}")
                        if face_detected:
                            st.write(f"**Coordenadas del rostro:** x={coords[0]}, y={coords[1]}, w={coords[2]}, h={coords[3]}")
                        
                        # Mostrar todas las probabilidades raw
                        st.write("**Todas las probabilidades:**")
                        all_probs = {labels.get(i, f"Clase {i}"): float(predictions[i]) * 100 
                                   for i in range(len(predictions))}
                        st.json(all_probs)
                        
                except Exception as e:
                    st.error(f"Error en la predicción: {e}")
                    st.exception(e)

if __name__ == "__main__":
    main()
