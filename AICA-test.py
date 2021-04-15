import os
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

@st.cache(allow_output_mutation=True)
def load():
    return load_model("./AICAv1.h5")

st.title("AICA Project - Prototype")
st.write("AICA: Artificial Intelligence Catching Alzheimer's")

model = load()
if model:
    st.write("Model Status: `Ready`")
else:
    st.error("Model Status: `No selected`")

input = st.file_uploader("Pick a MRI image file to predict", type=("png", "jpg"))

if input == None:
    st.warning("No file selected")
else:
    st.write("You selected `%s`" % input.name)
    st.image(input)

    a = Image.open(input)