
"""
Criado em Sab 02 de dezembro 18:31:28 2022
@autor: Marcelo Claro
Pacotes necessários: streamlit opencv-python Pillow numpy tensorflow
Modelo CNN: Covid19_CNN_Classifier.h5
"""


# Core Pkgs
import streamlit as st
st.set_page_config(page_title="Ferramenta de Detecção Covid-19", page_icon="covid19.jpg", layout='centered', initial_sidebar_state='auto')

import os
import time

# Viz Pkgs
import cv2
from PIL import Image,ImageEnhance
import numpy as np 

# AI Pkgs
import tensorflow as tf


def main():
	"""Ferramenta simples para detecção de Covid-19 por radiografia de tórax"""
	html_templ = """
	<div style="background-color:blue;padding:10px;">
	<h1 style="color:yellow">Detecção de Covid-19</h1>
	</div>
	"""

	st.markdown(html_templ,unsafe_allow_html=True)
	st.write("Uma proposta simples para o diagnóstico de Covid-19 com tecnologia Deep Learning e Streamlit")

	st.sidebar.image("covid19.jpg",width=300)

	image_file = st.sidebar.file_uploader("Carregar uma imagem de raio-X (jpg, png or jpeg)",type=['jpg','png','jpeg'])

	if image_file is not None:
		our_image = Image.open(image_file)

		if st.sidebar.button("Pré-visualização de imagem"):
			st.sidebar.image(our_image,width=300)

		activities = ["Melhoria de imagem","Diagnóstico", "Isenção de responsabilidade e informações"]
		choice = st.sidebar.selectbox("Selecione a atividade",activities)

		if choice == 'Melhoria de imagem':
			st.subheader("Melhoria de imagem")

			enhance_type = st.sidebar.radio("Melhorar tipo",["Original","Contraste","Brilho"])

			if enhance_type == 'Contraste':
				c_rate = st.slider("Contraste",0.5,5.0)
				enhancer = ImageEnhance.Contrast(our_image)
				img_output = enhancer.enhance(c_rate)
				st.image(img_output,use_column_width=True)


			elif enhance_type == 'Brilho':
				c_rate = st.slider("Brilho",0.5,5.0)
				enhancer = ImageEnhance.Brightness(our_image)
				img_output = enhancer.enhance(c_rate)
				st.image(img_output,width=600,use_column_width=True)


			else:
				st.text("Imagem Original")
				st.image(our_image,width=600,use_column_width=True)


		elif choice == 'Diagnóstico':
			
			if st.sidebar.button("Diagnóstico"):

				# Image to Black and White
				new_img = np.array(our_image.convert('RGB')) #our image is binary we have to convert it in array
				new_img = cv2.cvtColor(new_img,1) # 0 is original, 1 is grayscale
				gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
				st.text("Raio-x do tórax")
				st.image(gray,use_column_width=True)

				# PX-Ray (Image) Preprocessing
				IMG_SIZE = (200,200)
				img = cv2.equalizeHist(gray)
				img = cv2.resize(img,IMG_SIZE)
				img = img/255. #Normalization

				# Image reshaping according to Tensorflow format
				X_Ray = img.reshape(1,200,200,1)

				# Pre-Trained CNN Model Importing
				model = tf.keras.models.load_model('./models/Covid19_CNN_Classifier.h5')

				# Diagnóstico (Prevision=Binary Classification)
				Diagnóstico = model.predict_classes(X_Ray)
				Diagnóstico_proba = model.predict(X_Ray)
				probability_cov = Diagnóstico_proba*100
				probability_no_cov = (1-Diagnóstico_proba)*100

				my_bar = st.sidebar.progress(0)


				for percent_complete in range(100):
					time.sleep(0.05)
					my_bar.progress(percent_complete + 1)

				# Diagnóstico Cases: No-Covid=0, Covid=1
				if Diagnóstico == 0:
					st.sidebar.success("Diagnóstico: NO COVID-19 (Probability: %.2f%%)" % (probability_no_cov))
				else:
					st.sidebar.error("Diagnóstico: COVID-19 (Probability: %.2f%%)" % (probability_cov))

				st.warning("This Web App is just a DEMO about Artificial Neural Networks so there is no clinical value in its Diagnóstico and the author is not a Doctor!")


		else:
			st.subheader("Isenção de responsabilidade e informações")
			st.subheader("Disclaimer")
			st.write("**This Tool is just a DEMO about Artificial Neural Networks so there is no clinical value in its diagnosis and the author is not a Doctor!**")
			st.write("**Please don't take the diagnosis outcome seriously and NEVER consider it valid!!!**")
			st.subheader("Info")
			st.write("This Tool gets inspiration from the following works:")
			st.write("- [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)") 
			st.write("- [Fighting Corona Virus with Artificial Intelligence & Deep Learning](https://www.youtube.com/watch?v=_bDHOwASVS4)") 
			st.write("- [Deep Learning per la Diagnosi del COVID-19](https://www.youtube.com/watch?v=dpa8TFg1H_U&t=114s)")
			st.write("We used 206 Posterior-Anterior (PA) X-Ray [images](https://github.com/ieee8023/covid-chestxray-dataset/blob/master/metadata.csv) of patients infected by Covid-19 and 206 Posterior-Anterior X-Ray [images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) of healthy people to train a Convolutional Neural Network (made by about 5 million trainable parameters) in order to make a classification of pictures referring to infected and not-infected people.")
			st.write("Since dataset was quite small, some data augmentation techniques have been applied (rotation and brightness range). The result was quite good since we got 94.5% accuracy on the training set and 89.3% accuracy on the test set. Afterwards the model was tested using a new dataset of patients infected by pneumonia and in this case the performance was very good, only 2 cases in 206 were wrongly recognized. Last test was performed with 8 SARS X-Ray PA files, all these images have been classified as Covid-19.")
			st.write("Unfortunately in our test we got 5 cases of 'False Negative', patients classified as healthy that actually are infected by Covid-19. It's very easy to understand that these cases can be a huge issue.")
			st.write("The model is suffering of some limitations:")
			st.write("- small dataset (a bigger dataset for sure will help in improving performance)")
			st.write("- images coming only from the PA position")
			st.write("- a fine tuning activity is strongly suggested")
			st.write("")
			st.write("Anybody has interest in this project can drop me an email and I'll be very happy to reply and help.")


	if st.sidebar.button("About the Author"):
		st.sidebar.subheader("Ferramenta de teste para COVID-19")
		st.sidebar.markdown("Prof.Marcelo Claro")
		st.sidebar.markdown("marcelolcaro@geomaker.org")
		st.sidebar.text("All Rights Reserved (2022)")


if __name__ == '__main__':
		main()	