import cv2
import numpy as np
import tensorflow as tf
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import warnings
warnings.filterwarnings('ignore')

# Carregar o modelo
model = tf.keras.models.load_model('97 Sign Language ALS Classifier.h5')

# Definir as classes
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Função para pré-processar a imagem capturada
def preprocess_frame(frame):
    target_size = (224, 224)
    resized = cv2.resize(frame, target_size)
    processed = tf.keras.applications.efficientnet.preprocess_input(resized)
    processed = np.expand_dims(processed, axis=0)
    return processed

# Iniciar webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a webcam")
    exit()

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Espelhar a imagem (como um espelho)
    frame = cv2.flip(frame, 1)

    # Área de interesse (ROI) para detecção (ajuste conforme necessário)
    x1, y1, x2, y2 = 100, 100, 324, 324
    roi = frame[y1:y2, x1:x2]

    # Pré-processar e prever
    try:
        processed = preprocess_frame(roi)
        predictions = model.predict(processed)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index] * 100
        predicted_class = classes[predicted_class_index]

        # Exibir previsão na imagem
        label = f'{predicted_class} ({confidence:.2f}%)'
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Erro na predição:", e)

    # Desenhar retângulo da ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Mostrar imagem
    cv2.imshow('Classificação de Linguagem de Sinais', frame)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerrar
cap.release()
cv2.destroyAllWindows()
