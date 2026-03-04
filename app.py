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

def get_face(image, face_cascade):
    """Detecta rostro humano"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    
    if len(faces) == 0:
        return None
    
    # Tomar el más grande
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Verificar proporción humana
    if w/h < 0.6 or w/h > 1.5:
        return None
    
    # Recortar
    margin = int(0.2 * w)
    x1, y1 = max(0, x-margin), max(0, y-margin)
    x2, y2 = min(gray.shape[1], x+w+margin), min(gray.shape[0], y+h+margin)
    
    face = gray[y1:y2, x1:x2]
    face = cv2.resize(face, (640, 480))
    return face

def main():
    st.title("🎓 Sistema de Reconocimiento Facial")
    st.write("Identificación de compañeros")
    
    model, labels, face_cascade = load_model()
    if model is None:
        return
    
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
        
        if st.button("✨ IDENTIFICAR", type="primary", use_container_width=True):
            with st.spinner("Analizando..."):
                # 1. Detectar rostro (protección contra perros/objetos)
                face = get_face(img_array, face_cascade)
                
                if face is None:
                    st.error("❌ No se detectó un rostro humano")
                    st.write("Acércate a la cámara y asegúrate de que tu cara sea visible")
                    return
                
                # 2. Predecir (sin filtros, sin umbrales)
                face = face / 255.0
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1)
                
                preds = model.predict(face, verbose=0)[0]
                best_idx = int(np.argmax(preds))
                person_name = labels.get(best_idx, "Desconocido")
                
                # 3. Mostrar SIEMPRE el resultado (sin importar confianza)
                st.balloons()
                st.success("¡Identificación Completada!")
                
                st.markdown(f"""
                <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 20px 0;">
                    <h1 style="color: white; font-size: 48px; margin: 0;">
                        ✅ {person_name}
                    </h1>
                    <p style="color: white; font-size: 20px; margin-top: 10px;">
                        Miembro identificado
                    </p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
