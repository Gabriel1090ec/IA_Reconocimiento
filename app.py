import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

st.set_page_config(
    page_title="Reconocimiento Facial",
    page_icon="✅",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('mejor_modelo.h5')
        class_dict = np.load('etiquetas.npy', allow_pickle=True).item()
        labels = {int(v): str(k) for k, v in class_dict.items()}
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return model, labels, face_cascade
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

def detect_face_strict(image, face_cascade):
    """
    Detección estricta de rostros humanos.
    Retorna: (rostro_recortado, coordenadas, calidad)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Parámetros estrictos para evitar falsos positivos (perros, objetos)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,  # Más estricto (antes era 4)
        minSize=(100, 100)  # Tamaño mínimo más grande
    )
    
    if len(faces) == 0:
        return None, None, None
    
    # Tomar el rostro más grande y centrado
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Verificar que sea proporción de rostro humano (no tan alargado como un perro)
    aspect_ratio = w / h
    if aspect_ratio < 0.6 or aspect_ratio > 1.4:
        return None, None, None
    
    # Recortar con margen
    margin = int(0.2 * w)
    x1, y1 = max(0, x-margin), max(0, y-margin)
    x2, y2 = min(gray.shape[1], x+w+margin), min(gray.shape[0], y+h+margin)
    face_crop = gray[y1:y2, x1:x2]
    
    return face_crop, (x, y, w, h), aspect_ratio

def preprocess(face_crop):
    """Preprocesa el rostro detectado"""
    face_resized = cv2.resize(face_crop, (640, 480))
    face_norm = face_resized / 255.0
    face_norm = np.expand_dims(face_norm, axis=0)
    face_norm = np.expand_dims(face_norm, axis=-1)
    return face_norm

def main():
    st.title("🎓 Sistema de Reconocimiento Facial")
    st.write("Identificación de compañeros - Redes Neuronales")
    
    model, labels, face_cascade = load_model()
    if model is None:
        return
    
    # Interfaz limpia
    option = st.radio("", ["📷 Usar Cámara", "📁 Subir Foto"], horizontal=True)
    
    image = None
    if option == "📷 Usar Cámara":
        cam = st.camera_input("")
        if cam:
            image = Image.open(cam)
    else:
        file = st.file_uploader("", type=['jpg','jpeg','png'])
        if file:
            image = Image.open(file)
    
    if image is not None:
        img_array = np.array(image)
        st.image(image, use_column_width=True)
        
        if st.button("✨ IDENTIFICAR PERSONA", type="primary", use_container_width=True):
            with st.spinner("Analizando..."):
                try:
                    # PASO 1: Verificar que sea un rostro humano
                    face_crop, coords, aspect = detect_face_strict(img_array, face_cascade)
                    
                    if face_crop is None:
                        st.error("❌ No se detectó un rostro humano")
                        st.info("💡 Acércate a la cámara y asegúrate de que tu rostro sea visible")
                        return
                    
                    # PASO 2: Predecir
                    processed = preprocess(face_crop)
                    preds = model.predict(processed, verbose=0)[0]
                    
                    best_idx = int(np.argmax(preds))
                    confidence = float(preds[best_idx]) * 100
                    
                    # PASO 3: Verificar confianza (umbral oculto, no se muestra al usuario)
                    # Si es menos de 40%, probablemente no es ninguna persona registrada
                    if confidence < 40:
                        st.warning("⚠️ Persona no registrada")
                        st.write("No se encontró coincidencia en la base de datos")
                        return
                    
                    # PASO 4: Mostrar resultado seguro
                    person_name = labels.get(best_idx, "Desconocido")
                    
                    st.balloons()
                    st.success("¡Identificación Completada!")
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 20px 0;">
                        <h1 style="color: white; font-size: 48px; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                            ✅ {person_name}
                        </h1>
                        <p style="color: white; font-size: 20px; margin-top: 10px; opacity: 0.9;">
                            Miembro registrado
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error("Error en el proceso")
                    print(f"Error: {e}")

if __name__ == "__main__":
    main()
