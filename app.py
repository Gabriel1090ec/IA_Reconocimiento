import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

st.set_page_config(page_title="Reconocimiento Facial", layout="centered")

@st.cache_resource
def load_resources():
    if not os.path.exists('mejor_modelo.h5'):
        return None, None, None, "Error: No existe mejor_modelo.h5"
    if not os.path.exists('etiquetas.npy'):
        return None, None, None, "Error: No existe etiquetas.npy"
    
    try:
        model = tf.keras.models.load_model('mejor_modelo.h5')
        class_dict = np.load('etiquetas.npy', allow_pickle=True).item()
        labels = {int(v): str(k) for k, v in class_dict.items()}
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return model, labels, face_cascade, None
    except Exception as e:
        return None, None, None, f"Error cargando recursos: {str(e)}"

def extract_face(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))
    
    if len(faces) == 0:
        return None
    
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    if w/h < 0.6 or w/h > 1.4:
        return None
    
    margin = int(0.25 * w)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(gray.shape[1], x + w + margin)
    y2 = min(gray.shape[0], y + h + margin)
    
    face = gray[y1:y2, x1:x2]
    face_resized = cv2.resize(face, (640, 480))
    return face_resized

def predict(model, face_img):
    img_normalized = face_img.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    img_batch = np.expand_dims(img_batch, axis=-1)
    predictions = model.predict(img_batch, verbose=0)
    return predictions[0]

def main():
    st.title("Reconocimiento Facial")
    
    model, labels, face_cascade, error_msg = load_resources()
    
    if error_msg:
        st.error(error_msg)
        return
    
    option = st.radio("Metodo", ["Camara", "Subir Foto"], horizontal=True)
    
    image_file = None
    if option == "Camara":
        image_file = st.camera_input("Capturar")
    else:
        image_file = st.file_uploader("Seleccionar imagen", type=['jpg', 'jpeg', 'png'])
    
    if image_file is not None:
        image = Image.open(image_file).convert('RGB')
        img_array = np.array(image)
        
        st.image(image, use_column_width=True)
        
        if st.button("IDENTIFICAR", type="primary", use_container_width=True):
            face = extract_face(img_array, face_cascade)
            
            if face is None:
                st.error("No se detecto un rostro humano. Asegurate de estar frente a la camara.")
                return
            
            predictions = predict(model, face)
            best_idx = int(np.argmax(predictions))
            person_name = labels.get(best_idx, "Desconocido")
            
            st.success(f"{person_name}")

if __name__ == "__main__":
    main()
