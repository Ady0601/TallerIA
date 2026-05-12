from flask import Flask, render_template, request
import os
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog

app = Flask(__name__)

# Tu ID de Google Drive extraído del enlace
file_id = '1PVBcsGXl-HlAX_5EHe_lCJIUR-1FJCxv'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'taller_prendas.pkl'

# Solo descarga el modelo si no existe ya en el servidor
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
# Carga del modelo
modelo = joblib.load('taller_prendas.pkl')

nombres_clases = {
    '0': 'Camiseta (T-shirt/top)', '1': 'Pantalón (Trouser)', '2': 'Pullover (Saco)',
    '3': 'Vestido (Dress)', '4': 'Abrigo (Coat)', '5': 'Sandalia (Sandal)',
    '6': 'Camisa (Shirt)', '7': 'Zapatilla (Sneaker)', '8': 'Bolso (Bag)', '9': 'Botín (Ankle boot)'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            
            # Procesamiento con HOG (Fundamental para que coincida con el entrenamiento)
            img = Image.open(path).convert('L').resize((28, 28))
            img_array = np.array(img)
            
            features = hog(img_array, orientations=9, pixels_per_cell=(4, 4), 
                          cells_per_block=(2, 2), block_norm='L2-Hys')
            
            pred = modelo.predict(features.reshape(1, -1))
            resultado = nombres_clases[str(pred[0])]
            return render_template('index.html', prediccion=resultado)
            
    return render_template('index.html', prediccion=None)

if __name__ == '__main__':
    app.run(debug=True)