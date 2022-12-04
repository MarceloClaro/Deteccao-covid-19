
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

				st.warning("Este Web App é apenas uma DEMO sobre Redes Neurais Artificiais, portanto não há valor clínico em seu Diagnóstico e o autor não é Médico!")


		else:
			st.subheader("Isenção de responsabilidade e informações")
			st.subheader("Isenção de responsabilidade")
			st.write("**Esta ferramenta é apenas uma DEMO sobre Redes Neurais Artificiais, portanto não há valor clínico em seu diagnóstico e o autor não é médico!**")
			st.write("**Por favor, não leve a sério o resultado do diagnóstico e NUNCA o considere válido!!!**")
			st.subcabeçalho("Informações")
			st.write("Esta ferramenta foi inspirada nos seguintes trabalhos:")
			st.write("- [Detectando COVID-19 em imagens de raios X com Keras, TensorFlow e Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in- x-ray-images-with-keras-tensorflow-and-deep-learning/)")
			st.write("-[Combate ao vírus Corona com inteligência artificial e aprendizagem profunda](https://www.youtube.com/watch?v=_bDHOwASVS4)")
			st.write("- [Deep Learning per la Diagnosi del COVID-19](https://www.youtube.com/watch?v=dpa8TFg1H_U&t=114s)")
			st.write("Usamos 206 imagens de Raios-X Posterior-Anterior (PA)(https://github.com/ieee8023/covid-chestxray-
			dataset/blob/master/metadata.csv) de pacientes infectados por Covid-19 e 206 Posterior-Anterior X-Ray [imagens](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) de pessoas saudáveis ​​para treinar uma Rede Neural Convolucional (composta por cerca de 5 milhões de parâmetros treináveis) a fim de fazer uma classificação de quadros referentes a pessoas infectadas e não infectadas.")
			st.write("Como o conjunto de dados era muito pequeno, algumas técnicas de aumento de dados foram aplicadas (rotação e faixa de brilho). O resultado foi muito bom, pois obtivemos 94,5% de precisão no conjunto de treinamento e 89,3% de precisão no conjunto de teste. Posteriormente, o modelo foi testado usando um novo conjunto de dados de pacientes infectados por pneumonia e, neste caso, o desempenho foi muito bom, apenas 2 casos em 206 foram reconhecidos incorretamente. O último teste foi realizado com 8 arquivos SARS X-Ray PA, todas essas imagens foram classificadas como Covid-19.")
			st.write("Infelizmente em nosso teste obtivemos 5 casos de 'Falso Negativo', pacientes classificados como saudáveis ​​que na verdade estão infectados pelo Covid-19. É muito fácil entender que esses casos podem ser um grande problema.")
			st.write("O modelo está sofrendo de algumas limitações:")
			st.write("-pequeno conjunto de dados (um conjunto de dados maior com certeza ajudará a melhorar o desempenho)")
			st.write("- imagens vindas apenas da posição PA")
			st.write("- uma atividade de ajuste fino é fortemente sugerida")
			st.write("")
			st.write("Alguém que tenha interesse neste projeto pode me mandar um e-mail que terei o maior prazer em responder e ajudar.")


	if st.sidebar.button("Sobre o autor"):
		st.sidebar.subheader("Ferramenta de teste para COVID-19")
		st.sidebar.markdown("Prof.Marcelo Claro")
		st.sidebar.markdown("marcelolcaro@geomaker.org")
		st.sidebar.text("All Rights Reserved (2022)")


if __name__ == '__main__':
		main()	