# predict_image.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load model dan class indices
model_path = 'custom_model_sapi.h5'
model = load_model(model_path)  # Load model menggunakan load_model

# Upload gambar
# uploaded = files.upload()
# Jika file gambar berada di folder yang sama
uploaded_file = 'full.png'  # Ganti dengan nama file gambar yang ingin diprediksi

# Preprocessing gambar
img = image.load_img(uploaded_file, target_size=(64, 64), color_mode='rgb') # Load image as RGB
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Reshape the image array to match the expected input shape of the model
# img_array = img_array.reshape(1, 8192)  # Reshape to (1, 8192) for a single grayscale image

# Load the model
# model = load_model(model_path)  # Load the model using load_model

# Prediksi
prediction = model.predict(img_array) # Use the loaded model for prediction
predicted_class_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

# Tampilkan hasil
print(f"Hasil Prediksi: Kelas {predicted_class_index}")
print(f"Tingkat Kepercayaan: {confidence:.2f}%")

plt.imshow(img)
plt.title(f"Prediksi: Kelas {predicted_class_index} ({confidence:.2f}%)")
plt.axis('off')
plt.show()