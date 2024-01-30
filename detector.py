import cv2

def detect_faces(image_path):
    # Carregar o modelo pré-treinado
    modelo_caminho = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    classificador_rosto = cv2.CascadeClassifier(modelo_caminho)

    # Carregar a imagem
    imagem = cv2.imread(image_path)
    # Converter a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    rostos = classificador_rosto.detectMultiScale(
        imagem_cinza,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Recortar e retornar a imagem com o rosto
    for (x, y, w, h) in rostos:
        rosto_recortado = imagem[y:y+h, x:x+w]
        return rosto_recortado
