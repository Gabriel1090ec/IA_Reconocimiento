import streamlit as st
import numpy as np
import tensorflow as tf
tf.keras.config.enable_unsafe_deserialization()  
from PIL import Image
import cv2

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento Facial - Redes Neuronales",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Carga el modelo y las etiquetas una sola vez"""
    try:
        # Cargar modelo
        model = tf.keras.models.load_model('mejor_modelo.h5')
        
        # Cargar etiquetas desde .npy (formato original del entrenamiento)
        class_dict = np.load('etiquetas.npy', allow_pickle=True).item()
        
        # Invertir diccionario {id: nombre}
        labels = {v: k for k, v in class_dict.items()}
        
        return model, labels
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None

def preprocess_image(image):
    """Preprocesa la imagen para el modelo (640x480 grayscale)"""
    # Convertir a grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Redimensionar a 640x480 (width x height)
    image = cv2.resize(image, (640, 480))
    
    # Normalizar
    image = image / 255.0
    
    # Agregar dimensiones de batch y canal: (1, 480, 640, 1)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    return image

def main():
    st.title("🎓 Reconocimiento Facial - Compañeros")
    st.write("Sistema de reconocimiento usando Redes Neuronales (TensorFlow)")
    
    # Cargar modelo
    model, labels = load_model()
    
    if model is None:
        st.error("No se pudo cargar el modelo. Verifica que 'mejor_modelo.h5' y 'etiquetas.npy' existan.")
        return
    
    st.success(f"✅ Modelo cargado: {len(labels)} compañeros detectados")
    
    # Opciones de entrada
    option = st.radio("Selecciona método de entrada:", 
                      ["📷 Usar Cámara", "📁 Subir Imagen"])
    
    image_input = None
    
    if option == "📷 Usar Cámara":
        # Captura de cámara
        camera_image = st.camera_input("Toma una foto")
        if camera_image is not None:
            image_input = Image.open(camera_image)
    else:
        # Subir archivo
        uploaded_file = st.file_uploader("Elige una imagen...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image_input = Image.open(uploaded_file)
    
    if image_input is not None:
        # Mostrar imagen capturada
        st.image(image_input, caption="Imagen capturada", use_column_width=True)
        
        # Botón para analizar
        if st.button("🔍 Identificar Persona"):
            with st.spinner("Analizando..."):
                try:
                    # Convertir a array numpy
                    img_array = np.array(image_input)
                    
                    # Preprocesar
                    processed = preprocess_image(img_array)
                    
                    # Predecir
                    predictions = model.predict(processed, verbose=0)
                    class_id = np.argmax(predictions[0])
                    confidence = predictions[0][class_id] * 100
                    
                    # Obtener nombre
                    person_name = labels.get(class_id, "Desconocido")
                    
                    # Mostrar resultados
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("👤 Persona", person_name)
                    with col2:
                        st.metric("📊 Confianza", f"{confidence:.1f}%")
                    
                    # Barra de progreso para confianza
                    st.progress(int(confidence))
                    
                    # Alerta si confianza es baja
                    if confidence < 60:
                        st.warning("⚠️ Confianza baja - Acércate más a la cámara o mejora la iluminación")
                    else:
                        st.success("✅ Identificación exitosa")
                        
                except Exception as e:
                    st.error(f"Error en la predicción: {e}")

if __name__ == "__main__":
    main()
