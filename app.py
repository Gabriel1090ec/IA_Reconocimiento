import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

st.set_page_config(
    page_title="Reconocimiento Facial",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('mejor_modelo.h5')
        # Cargar etiquetas desde .npy
        class_dict = np.load('etiquetas.npy', allow_pickle=True).item()
        # Invertir: {0: "Juan", 1: "Maria"...}
        labels = {int(v): str(k) for k, v in class_dict.items()}
        
        # Cargar detector de rostros
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print(f"Etiquetas cargadas: {labels}")  # Debug
        return model, labels, face_cascade
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None, None

def detect_face(image, face_cascade):
    """Detecta y recorta el rostro"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Tomar el rostro más grande
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Margen del 20%
        margin = int(0.2 * w)
        x, y = max(0, x-margin), max(0, y-margin)
        w, h = min(gray.shape[1]-x, w + 2*margin), min(gray.shape[0]-y, h + 2*margin)
        face_crop = gray[y:y+h, x:x+w]
        return face_crop, True
    else:
        # Si no hay rostro, usar imagen completa en grayscale
        return gray, False

def preprocess(image, face_cascade):
    """Preprocesa la imagen para el modelo"""
    face_img, detected = detect_face(image, face_cascade)
    
    # Redimensionar a 640x480 (ancho x alto)
    face_img = cv2.resize(face_img, (640, 480))
    
    # Mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face_img = clahe.apply(face_img.astype(np.uint8))
    
    # Normalizar y reshape
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)  # Batch
    face_img = np.expand_dims(face_img, axis=-1) # Canal
    
    return face_img, detected

def main():
    st.title("🎓 Reconocimiento Facial - Compañeros")
    st.write("Redes Neuronales - TensorFlow")
    
    # Cargar modelo
    model, labels, face_cascade = load_model()
    
    if model is None:
        st.error("Error cargando el modelo")
        return
    
    st.success(f"✅ Sistema listo - {len(labels)} personas registradas")
    
    # Mostrar quiénes están registrados
    with st.expander("Ver personas registradas"):
        nombres = [labels[i] for i in sorted(labels.keys())]
        st.write(", ".join(nombres))
    
    # Entrada de imagen
    option = st.radio("Método:", ["📷 Cámara", "📁 Subir imagen"], horizontal=True)
    
    image = None
    if option == "📷 Cámara":
        cam = st.camera_input("Toma una foto")
        if cam:
            image = Image.open(cam)
    else:
        file = st.file_uploader("Selecciona imagen", type=['jpg','jpeg','png'])
        if file:
            image = Image.open(file)
    
    if image is not None:
        img_array = np.array(image)
        
        # Mostrar imagen original
        st.image(image, caption="Imagen capturada", use_column_width=True)
        
        if st.button("🔍 IDENTIFICAR AHORA", type="primary"):
            with st.spinner("Analizando..."):
                try:
                    # Preprocesar
                    processed, face_detected = preprocess(img_array, face_cascade)
                    
                    # Predecir
                    preds = model.predict(processed, verbose=0)[0]
                    
                    # Obtener el mejor resultado
                    best_idx = int(np.argmax(preds))
                    confidence = float(preds[best_idx]) * 100
                    
                    # OBTENER NOMBRE - Siempre muestra el resultado
                    person_name = labels.get(best_idx, f"Persona {best_idx}")
                    
                    # Mostrar resultado principal SIEMPRE (sin filtros)
                    st.divider()
                    st.subheader("🎯 RESULTADO")
                    
                    # Color según confianza (solo visual, no bloquea)
                    if confidence >= 80:
                        color = "green"
                        emoji = "✅"
                    elif confidence >= 50:
                        color = "orange" 
                        emoji = "⚠️"
                    else:
                        color = "red"
                        emoji = "❓"
                    
                    # Mostrar nombre GRANDE y claro
                    st.markdown(f"""
                    <div style="padding:20px; background-color:#f0f2f6; border-radius:10px; border-left:5px solid {color};">
                        <h1 style="margin:0; color:black; font-size:36px;">{emoji} {person_name}</h1>
                        <h3 style="margin:10px 0 0 0; color:{color};">Confianza: {confidence:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Barra de progreso
                    st.progress(int(confidence))
                    
                    # Detalles adicionales
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("ID de clase", best_idx)
                    with col2:
                        st.metric("Confianza raw", f"{confidence:.2f}%")
                    
                    # Mostrar top 3 siempre
                    st.divider()
                    st.write("**Top 3 posibilidades:**")
                    top3_idx = np.argsort(preds)[-3:][::-1]
                    for i, idx in enumerate(top3_idx, 1):
                        name = labels.get(int(idx), f"Clase {idx}")
                        conf = float(preds[idx]) * 100
                        st.write(f"{i}. {name}: {conf:.1f}%")
                    
                    # Info de debug (opcional)
                    with st.expander("Ver detalles técnicos"):
                        st.write(f"Rostro detectado: {'Sí' if face_detected else 'No'}")
                        st.write(f"Forma tensor: {processed.shape}")
                        st.write(f"Todas las clases: {labels}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)

if __name__ == "__main__":
    main()
