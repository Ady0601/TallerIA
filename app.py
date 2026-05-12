import os
import gdown
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from skimage import transform

app = Flask(__name__)

# ID de tu archivo en Google Drive (verificado de tu enlace)
file_id = '1PVBcsGXl-HlAX_5EHe_lCJIUR-1FJCxv'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'taller_prendas.pkl'

# Función para descargar el modelo si no existe
def download_model():
    if not os.path.exists(output):
        print("Descargando modelo desde Google Drive...")
        # fuzzy=True ayuda a encontrar el archivo incluso si el enlace cambia un poco
        gdown.download(url, output, quiet=False, fuzzy=True)

# Ejecutar la descarga antes de cargar el modelo
download_model()

# Cargar el modelo
try:
    modelo = joblib.load(output)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

def preprocess_image(image):
    # Convertir a escala de grises y redimensionar a 28x28 (como Fashion MNIST)
    img = image.convert('L')
    img = np.array(img)
    img = transform.resize(img, (28, 28), anti_aliasing=True)
    img = img.flatten() / 255.0  # Normalizar
    return img.reshape(1, -1)

@app.route('/')
def home():
    return "¡Servidor de Clasificación de Ropa funcionando!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se subió ningún archivo'}), 400
    
    file = request.files['file']
    img = Image.open(file)
    processed_img = preprocess_image(img)
    
    prediction = modelo.predict(processed_img)
    
    # Mapeo de categorías de Fashion MNIST (ajusta según tu modelo)
    categories = ['Camiseta/Top', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo', 
                  'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']
    
    result = categories[int(prediction[0])]
    return jsonify({'category': result})

if __name__ == '__main__':
    # Usar el puerto que asigne Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
