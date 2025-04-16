import cv2
import numpy as np
import tensorflow as tf
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import warnings
warnings.filterwarnings('ignore')

# Carregar o modelo
model = tf.keras.models.load_model('97 Sign Language ALS Classifier.h5')

# Definir as classes na ordem correta (mesma do treinamento)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Função para pré-processar a imagem
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Imagem não encontrada ou inválida: {image_path}")
    
    target_size = (224, 224)
    resized = cv2.resize(img, target_size)
    processed = tf.keras.applications.efficientnet.preprocess_input(resized)
    processed = np.expand_dims(processed, axis=0)
    return processed, resized

# Caminho da imagem
image_path = 'content.png'

# Pré-processar
processed_image, _ = preprocess_image(image_path)

# Fazer previsão
predictions = model.predict(processed_image)
predicted_class_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_class_index] * 100

# Obter a letra correspondente usando a nova ordem de classes
predicted_class = classes[predicted_class_index]

# Exibir resultado com depuração
print("Previsões brutas (probabilidades para cada classe):", predictions[0])
print("Classe prevista (índice):", predicted_class_index)
print("Todas as classes:", classes)
print("Confiança em todas as classes (%):", [f"{p*100:.2f}" for p in predictions[0]])
print(f'Letra prevista: {predicted_class} (Confiança: {confidence:.2f}%)')
