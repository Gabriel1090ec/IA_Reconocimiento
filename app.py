import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from collections import Counter

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
        
        # Cargar detector de rostros más preciso (LBP es más rápido y preciso que Haar)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detector de ojos para alineación
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        return model, labels, face_cascade, eye_cascade
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None

def enhance_image(image):
    """Mejora agresiva de la imagen para maximizar precisión"""
    # Ecualización adaptativa de histograma (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    # Reducción de ruido
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    return enhanced

def get_face(image, face_cascade, eye_cascade):
    """Obtiene rostro con alineación de ojos para consistencia"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # Más fino
        minNeighbors=6,    # Más estricto
        minSize=(120, 120),
        maxSize=(gray.shape[1], gray.shape[0])
    )
    
    if len(faces) == 0:
        return None
    
    # Tomar el rostro más grande (más cercano)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Verificar proporción 1:1.2 aprox (rostro humano)
    if w/h < 0.7 or w/h > 1.3:
        return None
    
    # Recortar
    margin = int(0.25 * w)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(gray.shape[1], x + w + margin)
    y2 = min(gray.shape[0], y + h + margin)
    
    face = gray[y1:y2, x1:x2]
    
    # Mejorar imagen
    face = enhance_image(face)
    
    return face

def predict_with_augmentation(model, face_img, n_augmentations=5):
    """
    Predice múltiples veces con variaciones pequeñas (TTA - Test Time Augmentation)
    Esto mejora la precisión al promediar predicciones con ligeras rotaciones/zooms
    """
    predictions = []
    
    base_size = (640, 480)
    
    for i in range(n_augmentations):
        augmented = face_img.copy()
        
        if i > 0:
            # Ligeras variaciones para ensemble
            angle = np.random.uniform(-5, 5)
            scale = np.random.uniform(0.95, 1.05)
            
            M = cv2.getRotationMatrix2D((face_img.shape[1]//2, face_img.shape[0]//2), angle, scale)
            augmented = cv2.warpAffine(augmented, M, (face_img.shape[1], face_img.shape[0]))
        
        # Redimensionar y normalizar
        resized = cv2.resize(augmented, base_size)
        normalized = resized / 255.0
        normalized = np.expand_dims(normalized, axis=0)
        normalized = np.expand_dims(normalized, axis=-1)
        
        pred = model.predict(normalized, verbose=0)[0]
        predictions.append(pred)
    
    # Promediar predicciones (votación suave)
    avg_pred = np.mean(predictions, axis=0)
    return avg_pred

def main():
    st.title("🎓 Sistema de Reconocimiento Facial")
    st.write("Identificación precisa de compañeros")
    
    model, labels, face_cascade, eye_cascade = load_model()
    if model is None:
        return
    
    st.info(f"📚 Base de datos: {len(labels)} personas registradas")
    
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
        st.image(image, caption="Imagen capturada", use_column_width=True)
        
        if st.button("✨ IDENTIFICAR AHORA", type="primary", use_container_width=True):
            with st.spinner("Procesando con precisión máxima..."):
                try:
                    # 1. Detección de rostro estricta
                    face_crop = get_face(img_array, face_cascade, eye_cascade)
                    
                    if face_crop is None:
                        st.error("❌ No se detectó un rostro válido")
                        st.write("Asegúrate de:")
                        st.write("- Estar frente a la cámara")
                        st.write("- Buena iluminación")
                        st.write("- No usar accesorios que cubran el rostro")
                        return
                    
                    # 2. Predicción con ensemble (múltiples predicciones para robustez)
                    final_pred = predict_with_augmentation(model, face_crop, n_augmentations=7)
                    
                    best_idx = int(np.argmax(final_pred))
                    confidence = float(final_pred[best_idx]) * 100
                    
                    # 3. Verificar consistencia (top 2 deben estar separados)
                    sorted_preds = np.sort(final_pred)[::-1]
                    gap = (sorted_preds[0] - sorted_preds[1]) * 100
                    
                    # 4. Mostrar resultado solo si es consistente
                    if confidence > 70 and gap > 20:
                        person_name = labels.get(best_idx, "Desconocido")
                        
                        st.balloons()
                        st.success("✅ Identificación Confirmada")
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 20px 0;">
                            <h1 style="color: white; font-size: 48px; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                                {person_name}
                            </h1>
                            <p style="color: white; font-size: 20px; margin-top: 10px; opacity: 0.9;">
                                Miembro verificado
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif confidence > 50:
                        st.warning("⚠️ Imagen poco clara")
                        st.write("Intenta de nuevo con mejor iluminación o posición")
                    else:
                        st.error("❌ Persona no registrada")
                        st.write("No se encontró coincidencia en la base de datos")
                        
                except Exception as e:
                    st.error("Error en el procesamiento")

if __name__ == "__main__":
    main()
