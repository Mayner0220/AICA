import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

CLASS_NAMES = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']

@st.cache(allow_output_mutation=True)
def load():
    return load_model("./AICAv1.h5")

def file_selector(folder_path='./'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

st.title("AICA Project - Prototype")
st.write("AICA: Artificial Intelligence Catching Alzheimer's")

model = load()
if model:
    st.write("Model Status: `Ready`")
else:
    st.error("Model Status: `No selected`")

# input = st.file_uploader("Pick a MRI image file to predict", type=("png", "jpg"))
file_path = st.text_input("Please enter the path of the image to be predicted", "./")

if file_path == None:
    st.warning("No file selected")
else:
    if os.path.isfile(file_path):
        st.write("You selected `%s`" % file_path)

        image = tf.keras.preprocessing.image.load_img(file_path, target_size=(208, 176))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.expand_dims(image, 0)

        predictions = model.predict(image)
        score = tf.nn.softmax(predictions[0])

        raw_image = Image.open(file_path)
        st.image(raw_image, use_column_width="auto")
        st.write("Predict result: {}".format(CLASS_NAMES[np.argmax(score)]))
        st.write("Score: {:.2f}".format(100 * np.max(score)))
    else:
        st.warning("There is no file")