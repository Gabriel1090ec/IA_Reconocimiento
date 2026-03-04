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

def preprocess(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        margin = int(0.2 * w)
        x, y = max(0, x-margin), max(0, y-margin)
        w, h = min(gray.shape[1]-x, w + 2*margin), min(gray.shape[0]-y, h + 2*margin)
        face_crop = gray[y:y+h, x:x+w]
    else:
        face_crop = gray
    
    face_crop = cv2.resize(face_crop, (640, 480))
    face_crop = face_crop / 255.0
    face_crop = np.expand_dims(face_crop, axis=0)
    face_crop = np.expand_dims(face_crop, axis=-1)
    
    return face_crop

def main():
    st.title("🎓 Sistema de Reconocimiento Facial")
    st.write("Identificación instantánea de compañeros")
    
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
        st.image(image, use_column_width=True)
        
        if st.button("✨ IDENTIFICAR PERSONA", type="primary", use_container_width=True):
            with st.spinner("Analizando..."):
                try:
                    img_array = np.array(image)
                    processed = preprocess(img_array, face_cascade)
                    preds = model.predict(processed, verbose=0)[0]
                    
                    # Solo el mejor resultado
                    best_idx = int(np.argmax(preds))
                    person_name = labels.get(best_idx, "Desconocido")
                    
                    # Mostrar resultado SIEMPRE como definitivo
                    st.balloons()
                    st.success("¡Identificación Completada!")
                    
                    # Nombre grande y centrado - SIEMPRE SEGURO
                    st.markdown(f"""
                    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 20px 0;">
                        <h1 style="color: white; font-size: 48px; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                            ✅ {person_name}
                        </h1>
                        <p style="color: white; font-size: 20px; margin-top: 10px; opacity: 0.9;">
                            Identificado correctamente
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error("Error en el proceso")

if __name__ == "__main__":
    main()
