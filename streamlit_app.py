import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Flatten, Dropout, Dense
from tensorflow.keras import regularizers
import numpy as np
import wget
import zipfile
import os
import contextlib

st.write(f"TensorFlow version: {tf.__version__}")
# Descargar y descomprimir el modelo si no existe
def download_and_extract_model():
    model_url = 'https://dl.dropboxusercontent.com/s/sdqx2xu119uqacd12f5u3/Xception_model.zip?rlkey=9qap8uwhag1f8nmo87y51ypaa&st=shs6bi0n'
    zip_path = 'Xception_model.zip  '
    extract_folder = 'extracted_files'

    # Descargar el archivo zip si no existe
    if not os.path.exists(zip_path):
        try:
            wget.download(model_url, zip_path)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")
            return False

    # Descomprimir el archivo
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    return os.path.join(extract_folder, 'Xception_model.keras')

modelo_path = download_and_extract_model()

# Verificar si el archivo del modelo existe
if not modelo_path or not os.path.exists(modelo_path):
    st.error("No se encontró el archivo del modelo")
else:
    st.success("Archivo del modelo encontrado")

# Definir el modelo base Xception
base_model =  Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
base_model.trainable = False

# Añadir capas de clasificación
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(2, activation='sigmoid')  # Cambiado a sigmoide para salida binaria
])

# Cargar los pesos del modelo desde el archivo .keras
try:
    model.load_weights(modelo_path)
    st.success("Pesos del modelo cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar los pesos del modelo: {e}")

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded_file is not None and model is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, width=300, caption="Imagen cargada")

    # Preprocesamiento de la imagen para hacer la predicción
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realizar la predicción con redirección de salida para evitar UnicodeEncodeError
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            prediction = model.predict(img_array)

    # Mostrar resultados
    if prediction[0][0] > 0.5:
        st.success('Prediccion: No existe presencia de neumonia.')
    else:
        st.success('Prediccion: Presencia de neumonia.')