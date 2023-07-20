import tensorflow as tf
import streamlit as st
from annotated_text import annotated_text, annotation
from PIL import Image
import matplotlib.pyplot as plt
st.set_page_config(page_title='Cat and Dog Classifier', page_icon='logo.png')

col1, col2 = st.columns(2)
col2.title('CAT AND DOG CLASSIFIER')
st.write('By Minh Tuan Lam')
col1.image('catndog.png')
uploaded_file = st.file_uploader('Choose an image to classify', type=['png', 'jpg'])
test = [93, 7]
colors = ['#CBFFA9', '#EF6262']
classes = ['Cat', 'Dog']
explode = [0.05, 0.05]
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img)
    col4, col5= st.columns(2)
    
    with col4:
        fig, axs = plt.subplots()
        axs.pie(test, colors=colors, labels=classes, explode=explode, autopct='%1.1f%%')
        st.pyplot(fig)
    with col5:
        annotated_text(annotation('There is a CAT in the image!', font_family='Comic Sans MS', border='2px dashed red', font_size='25px', margin='auto'))