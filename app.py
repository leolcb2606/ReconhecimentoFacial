from flask import Flask, render_template, request, jsonify
from detector import detect_faces
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    return render_template('index.html', result_path=None)

@app.route('/detect_faces', methods=['POST'])
def detect_faces_route():
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"})

    image = request.files['image']

    if image.filename == '':
        return jsonify({"error": "No selected file"})

    # Salvar a imagem no diretório de uploads
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user_image.jpg')
    image.save(image_path)

    # Executar o algoritmo de detecção facial
    rosto_recortado = detect_faces(image_path)

    # Salvar a imagem com o rosto recortado
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
    cv2.imwrite(output_path, rosto_recortado)

    # Retornar o caminho da imagem com o rosto recortado
    return render_template('index.html', result_path='/static/uploads/output.jpg')

if __name__ == '__main__':
    app.run(debug=True)
