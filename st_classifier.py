import tensorflow as tf
import streamlit as st
from annotated_text import annotated_text, annotation
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
st.set_page_config(page_title='Cat and Dog Classifier', page_icon='logo.png')
col1, col2 = st.columns(2)
col2.title('CAT AND DOG CLASSIFIER')
st.write('By Minh Tuan Lam')
col1.image('catndog.png')
uploaded_file = st.file_uploader('Choose an image to classify', type=['png', 'jpg'])
colors = ['#CBFFA9', '#EF6262']
classes = ['Cat', 'Dog']
explode = [0.05, 0.05]
multipliers = np.array([100, 100])
model = tf.keras.models.load_model('classifier')
if uploaded_file is not None:
    ori_img = Image.open(uploaded_file)
    st.image(ori_img)
    img = ori_img.resize((224, 224))
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    col4, col5= st.columns(2)
    with col4:
        pred = pred.flatten()
        pred = np.multiply(pred, multipliers)
        lb = np.argmax(pred)
        fig, axs = plt.subplots()
        axs.pie(pred, colors=colors, labels=classes, explode=explode, autopct='%1.1f%%')
        st.pyplot(fig)
    with col5:
        if lb == 0:
            annotated_text(annotation('There is a CAT in the image!', font_family='Comic Sans MS', border='2px dashed red', font_size='25px', margin='auto'))
        else:
            annotated_text(annotation('There is a DOG in the image!', font_family='Comic Sans MS', border='2px dashed red', font_size='25px', margin='auto'))