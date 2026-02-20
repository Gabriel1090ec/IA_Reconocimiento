import cv2
import os
import numpy as np

def train_face_recognizer(data_path='ITSE', model_path='modelo_entrenado.xml', label_map_path='etiquetas_personas.npy'):
    """
    Entrena un modelo LBPH para reconocimiento facial.
    
    Args:
        data_path: Ruta a la carpeta con subcarpetas por persona
        model_path: Ruta de salida para el modelo entrenado
        label_map_path: Ruta de salida para el mapeo ID->nombre
    
    Returns:
        tuple: (éxito: bool, estadísticas: dict)
    """
    # Validar existencia de datos
    if not os.path.exists(data_path):
        print(f"Error: Directorio '{data_path}' no encontrado")
        return False, {}
    
    # Cargar clasificadores Haar
    frontal_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml'
    )
    
    # Obtener lista de personas (solo directorios)
    people = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])
    
    if not people:
        print(f"Error: No se encontraron subdirectorios en '{data_path}'")
        return False, {}
    
    # Estructuras para almacenamiento
    labels = []
    faces = []
    label_map = {}
    
    # Procesar cada persona
    for label_id, person_name in enumerate(people):
        person_path = os.path.join(data_path, person_name)
        image_files = [
            f for f in os.listdir(person_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        
        if not image_files:
            continue
        
        face_count = 0
        
        for filename in image_files:
            # Leer imagen en escala de grises
            img_path = os.path.join(person_path, filename)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if gray is None:
                continue
            
            height, width = gray.shape
            
            # Estrategia adaptativa según tamaño de imagen:
            # - Imágenes grandes (>80px): detectar rostro con Haar Cascade
            # - Imágenes pequeñas (≤80px): asumir que toda la imagen es el rostro (ya recortada manualmente)
            if min(width, height) > 80:
                # Intentar detección frontal
                faces_rect = frontal_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )
                
                # Si falla, intentar detección de perfil
                if len(faces_rect) == 0:
                    faces_rect = profile_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                    )
                
                if len(faces_rect) == 0:
                    continue
                
                x, y, w, h = faces_rect[0]
                face_roi = gray[y:y+h, x:x+w]
            else:
                # Imagen pequeña: asumir que ya está recortada al rostro
                face_roi = gray
            
            # Normalizar tamaño a 150x150
            face_normalized = cv2.resize(
                face_roi, (150, 150), interpolation=cv2.INTER_AREA
            )
            
            # 🔑 NORMALIZACIÓN DE ILUMINACIÓN (mejora crítica de consistencia)
            face_normalized = cv2.equalizeHist(face_normalized)
            
            labels.append(label_id)
            faces.append(face_normalized)
            face_count += 1
        
        if face_count > 0:
            label_map[label_id] = person_name
    
    # Validar datos suficientes para entrenamiento
    if len(label_map) < 2:
        print("Error: Se requieren mínimo 2 personas con imágenes válidas para entrenar")
        return False, {}
    
    # Entrenar modelo LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )
    
    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)
    np.save(label_map_path, label_map)
    
    # Generar estadísticas
    stats = {
        'total_personas': len(label_map),
        'total_muestras': len(faces),
        'personas_procesadas': list(label_map.values()),
        'distribucion': {name: labels.count(lid) for lid, name in label_map.items()}
    }
    
    return True, stats


def main():
    print("Sistema de Entrenamiento de Reconocimiento Facial")
    print("=" * 60)
    
    success, stats = train_face_recognizer()
    
    if not success:
        return
    
    print("\nResumen de Entrenamiento")
    print("-" * 60)
    print(f"Personas procesadas : {stats['total_personas']}")
    print(f"Muestras totales     : {stats['total_muestras']}")
    print(f"\nDistribución por persona:")
    
    for name, count in sorted(stats['distribucion'].items()):
        print(f"  {name:20s} : {count:3d} muestras")
    
    # 🔍 Advertencia opcional sobre desbalance (útil para diagnóstico)
    counts = list(stats['distribucion'].values())
    if max(counts) > min(counts) * 3:
        print(f"\n⚠️  Nota: Dataset desbalanceado ({min(counts)} a {max(counts)} muestras por persona)")
        print("   Esto puede afectar la precisión en personas con pocas muestras.")
    
    print("\nModelo generado exitosamente:")
    print("  - modelo_entrenado.xml")
    print("  - etiquetas_personas.npy")
    print("=" * 60)


if __name__ == '__main__':
    main()
