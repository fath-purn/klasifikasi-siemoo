from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from io import BytesIO
import os

app = Flask(__name__)

model_path = 'custom_model_sapi.h5'
model = load_model(model_path)  # Load model menggunakan load_model

# Definisi kelas manual (sesuaikan dengan model Anda)
CLASS_INFO = {
    0: {
        "penyakit": "FMD (Foot and Mouth Disease)",
        "saran": "Isolasi hewan yang terinfeksi, vaksinasi rutin, dan desinfeksi kandang.",
        "bahaya": 3,
        "deskripsi": "Penyakit menular yang disebabkan oleh virus. Menyerang hewan berkuku belah seperti sapi, kerbau, domba, dan kambing. Gejala: demam, lepuh di mulut dan kaki, produksi susu menurun.",
    },
    1: {
        "penyakit": "IBK (Infectious Bovine Keratoconjunctivitis)",
        "saran": "Pisahkan hewan yang terinfeksi, berikan antibiotik, dan jaga kebersihan kandang.",
        "bahaya": 2,
        "deskripsi": "Penyakit mata yang disebabkan oleh bakteri. Dikenal juga sebagai 'pink eye' pada sapi. Gejala: mata merah, bengkak, keluar air mata berlebihan, dan kebutaan jika parah.",
    },
    2: {
        "penyakit": "LSD (Lumpy Skin Disease)",
        "saran": "Vaksinasi, isolasi hewan yang terinfeksi, dan kontrol vektor (serangga penular).",
        "bahaya": 3,
        "deskripsi": "Penyakit kulit yang disebabkan oleh virus. Menyebabkan benjolan-benjolan pada kulit sapi. Gejala: demam, benjolan di kulit, penurunan berat badan, dan produksi susu menurun.",
    }
}

def preprocess_image_from_url(url):
    # Download gambar dari URL
    response = requests.get(url)
    img = image.load_img(BytesIO(response.content), 
                        target_size=(64, 64),
                        color_mode='rgb')
    
    # Preprocessing
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get URL dari request body
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'Request body kosong',
                'data': None
            }), 400
            
        image_url = data.get('image_url')

        if not isinstance(image_url, str):
            return jsonify({
                'success': False,
                'message': 'URL gambar tidak valid',
                'data': None
            }), 400
            
        if not image_url.startswith(('http://', 'https://')):
            return jsonify({
                'success': False,
                'message': 'URL gambar tidak valid',
                'data': None
            }), 400
        
        if not image_url:
            return jsonify({
                'success': False,
                'message': 'URL gambar tidak valid',
                'data': None
            }), 400
        
        # Preprocess gambar
        img_array = preprocess_image_from_url(image_url)
        
        # Prediksi
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        
        # Get class info
        class_info = CLASS_INFO.get(predicted_class_index, {})
        
        return jsonify({
            'success': True,
            'message': 'Berhasil mendapatkan data',
            'data': {
                'akurasi': confidence,
                'penyakit': class_info.get('penyakit', 'Unknown'),
                'saran': class_info.get('saran', 'Unknown'),
                'bahaya': class_info.get('bahaya', 'Unknown'),
                'deskripsi': class_info.get('deskripsi', 'Unknown'),
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500

@app.route('/', methods=['GET'])
def home():
    try:
        # Baca kode dari file .env
        kode = os.getenv('kode')
        response = {
            'success': True,
            'message': 'Berhasil mendapatkan data',
            'data': 'Selamat datang',
            'kode': kode
        }
        return jsonify(response), 200
    except Exception as e:
        response = {
            'success': False, 
            'message': str(e),
            'data': None
        }
        return jsonify(response), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
