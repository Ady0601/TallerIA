import os
import gdown
import joblib
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from skimage import transform

app = Flask(__name__)

# Configuración de Google Drive (Tu ID de archivo público)
file_id = '1PVBcsGXl-HlAX_5EHe_lCJIUR-1FJCxv'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'taller_prendas.pkl'

# Descarga el modelo si no está presente (Esto evita el error de los 25MB de GitHub)
if not os.path.exists(output):
    print("Descargando modelo...")
    gdown.download(url, output, quiet=False, fuzzy=True)

# Carga el modelo original
modelo = joblib.load(output)

def preprocess_image(image):
    # Lógica original que funcionaba
    img = image.convert('L')
    img = np.array(img)
    img = transform.resize(img, (28, 28), anti_aliasing=True)
    img = img.flatten() / 255.0
    return img.reshape(1, -1)

@app.route('/')
def home():
    return "¡Servidor de Clasificación de Ropa funcionando correctamente!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió ninguna imagen'}), 400
    
    file = request.files['file']
    img = Image.open(file)
    processed_img = preprocess_image(img)
    
    prediction = modelo.predict(processed_img)
    
    # Categorías de Fashion MNIST
    categories = ['Camiseta/Top', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo', 
                  'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']
    
    result = categories[int(prediction[0])]
    return jsonify({'category': result})

if __name__ == '__main__':
    # Configuración de puerto para Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
