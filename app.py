import os
import gdown
import joblib
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from skimage import transform

app = Flask(__name__)

# ID de tu archivo en Google Drive (verificado de tu enlace público)
file_id = '1PVBcsGXl-HlAX_5EHe_lCJIUR-1FJCxv'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'taller_prendas.pkl'

def download_model():
    """Descarga el modelo desde Google Drive si no existe localmente."""
    if not os.path.exists(output):
        print(f"Iniciando descarga del modelo: {output}...")
        try:
            # Usamos fuzzy=True para manejar redirecciones de archivos grandes
            gdown.download(url, output, quiet=False, fuzzy=True)
            if os.path.exists(output):
                print("Descarga completada exitosamente.")
            else:
                print("Error: El archivo no se descargó correctamente.")
        except Exception as e:
            print(f"Error crítico durante la descarga: {e}")

# Intentar descargar y cargar el modelo al arrancar
download_model()

modelo = None
if os.path.exists(output):
    try:
        modelo = joblib.load(output)
        print("Modelo cargado en memoria correctamente.")
    except Exception as e:
        print(f"Error al cargar el archivo .pkl: {e}")
else:
    print("ALERTA: El archivo del modelo no existe. Las predicciones fallarán.")

def preprocess_image(image):
    """Preprocesa la imagen para que coincida con el formato Fashion MNIST (28x28)."""
    # Convertir a escala de grises
    img = image.convert('L')
    img = np.array(img)
    # Redimensionar a 28x28
    img = transform.resize(img, (28, 28), anti_aliasing=True)
    # Normalizar y aplanar
    img = img.flatten() / 255.0
    return img.reshape(1, -1)

@app.route('/')
def home():
    status = "Cargado" if modelo else "No cargado (revisar logs)"
    return f"¡Servidor de Clasificación de Ropa funcionando! Estado del modelo: {status}"

@app.route('/predict', methods=['POST'])
def predict():
    if modelo is None:
        return jsonify({'error': 'El modelo no está disponible en el servidor'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió ninguna imagen'}), 400
    
    try:
        file = request.files['file']
        img = Image.open(file)
        processed_img = preprocess_image(img)
        
        prediction = modelo.predict(processed_img)
        
        # Categorías estándar de Fashion MNIST
        categories = [
            'Camiseta/Top', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo', 
            'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín'
        ]
        
        result = categories[int(prediction[0])]
        return jsonify({'category': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render asigna dinámicamente un puerto a través de la variable de entorno PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
