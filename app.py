import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

st.set_page_config(page_title="Reconocimiento Facial", layout="centered")

@st.cache_resource
def cargar_recursos():
    """Carga modelo y etiquetas con verificación completa"""
    errores = []
    
    if not os.path.exists('mejor_modelo.h5'):
        errores.append("Falta archivo: mejor_modelo.h5")
    if not os.path.exists('etiquetas.npy'):
        errores.append("Falta archivo: etiquetas.npy")
    
    if errores:
        return None, None, " | ".join(errores)
    
    try:
        # Cargar modelo
        modelo = tf.keras.models.load_model('mejor_modelo.h5')
        
        # Verificar input shape del modelo
        input_shape = modelo.input_shape
        if input_shape != (None, 480, 640, 1):
            return None, None, f"Error: El modelo espera input {input_shape}, no (None, 480, 640, 1)"
        
        # Cargar etiquetas
        dict_clases = np.load('etiquetas.npy', allow_pickle=True).item()
        
        # Verificar que sea diccionario válido
        if not isinstance(dict_clases, dict):
            return None, None, "Error: etiquetas.npy no contiene un diccionario válido"
        
        # Invertir diccionario: {0: 'Juan', 1: 'Maria'}
        etiquetas = {int(v): str(k) for k, v in dict_clases.items()}
        
        return modelo, etiquetas, None
        
    except Exception as e:
        return None, None, f"Error carga: {str(e)}"

def preprocesar(imagen_pil):
    """Preprocesamiento exacto según entrenamiento"""
    # Convertir a array numpy
    img_array = np.array(imagen_pil.convert('RGB'))
    
    # Convertir a grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Redimensionar EXACTAMENTE a 640x480 (ancho x alto)
    # Tu modelo fue entrenado con 640 de ancho y 480 de alto
    procesada = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_AREA)
    
    # Normalizar igual que en entrenamiento (rescale=1./255)
    procesada = procesada.astype(np.float32) / 255.0
    
    # Agregar dimensiones: batch (1) + canal (1) = shape (1, 480, 640, 1)
    procesada = np.expand_dims(procesada, axis=0)  # Batch
    procesada = np.expand_dims(procesada, axis=-1) # Canal
    
    return procesada

def predecir(modelo, imagen_tensor):
    """Realiza predicción y retorna top 3 para verificación"""
    try:
        prediccion = modelo.predict(imagen_tensor, verbose=0)
        probabilidades = prediccion[0]
        
        # Obtener top 3
        indices_top = np.argsort(probabilidades)[-3:][::-1]
        confianzas = [float(probabilidades[i]) * 100 for i in indices_top]
        
        return int(indices_top[0]), confianzas, indices_top
    except Exception as e:
        raise Exception(f"Error en predicción: {str(e)}")

def main():
    st.title("Sistema de Reconocimiento Facial")
    st.caption("Redes Neuronales - TensorFlow")
    
    # Cargar recursos
    modelo, etiquetas, error = cargar_recursos()
    
    if error:
        st.error(error)
        st.info("Asegúrate de tener en el mismo directorio: mejor_modelo.h5 y etiquetas.npy")
        return
    
    # Mostrar info del modelo cargado
    with st.expander("Verificar carga del modelo"):
        st.write(f"Total de personas registradas: {len(etiquetas)}")
        st.write("Personas:", list(etiquetas.values()))
    
    # Interfaz
    metodo = st.radio("Seleccionar método", ["Cámara", "Subir archivo"], horizontal=True)
    
    archivo_imagen = None
    if metodo == "Cámara":
        archivo_imagen = st.camera_input("Capturar imagen")
    else:
        archivo_imagen = st.file_uploader("Seleccionar imagen JPG/PNG", type=['jpg','jpeg','png'])
    
    if archivo_imagen is not None:
        try:
            # Cargar imagen
            imagen = Image.open(archivo_imagen)
            st.image(imagen, caption="Imagen cargada", use_column_width=True)
            
            # Botón de identificación
            if st.button("IDENTIFICAR PERSONA", type="primary", use_container_width=True):
                with st.spinner("Procesando..."):
                    
                    # Preprocesar
                    tensor = preprocesar(imagen)
                    
                    # Verificar shape
                    if tensor.shape != (1, 480, 640, 1):
                        st.error(f"Error de preprocesamiento: Shape resultante {tensor.shape} != (1, 480, 640, 1)")
                        return
                    
                    # Predecir
                    clase_id, confianzas, top_indices = predecir(modelo, tensor)
                    
                    # Obtener nombre
                    nombre = etiquetas.get(clase_id, f"ID_{clase_id}")
                    
                    # Mostrar resultado principal
                    st.success(f"{nombre}")
                    
                    # Mostrar detalles de confianza (para verificar que no sea aleatorio)
                    st.divider()
                    st.caption("Detalles de predicción (debug):")
                    
                    datos_top = []
                    for idx, id_clase in enumerate(top_indices):
                        nom = etiquetas.get(id_clase, f"ID_{id_clase}")
                        conf = confianzas[idx]
                        datos_top.append(f"{nom}: {conf:.1f}%")
                    
                    st.write(" | ".join(datos_top))
                    
                    # Alerta si la confianza es muy baja (menos de 30%)
                    if confianzas[0] < 30:
                        st.warning("Precaución: Baja confianza en la predicción")
                    
        except Exception as e:
            st.error(f"Error procesando imagen: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
