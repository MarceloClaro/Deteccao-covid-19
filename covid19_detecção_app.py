
"""
Criado em 03/12/2022
@autor: Marcelo Claro
Pacotes Necessários: streamlit opencv-python Pillow numpy tensorflow
Modelo CNN: Covid19_CNN_Classifier.h5
"""


# Pacotes principais para a aplicação web e para a detecção de COVID-19 em imagens de raio-X do tórax 
import streamlit as st # Interface web	
st.set_page_config(page_title="Ferramenta de Detecção Covid19", page_icon="covid19.jpg", layout='centered', initial_sidebar_state='auto') 

import os # Para manipulação de arquivos e diretórios 
import time # Para manipulação de tempo 

# Viz Pkgs # Para visualização de imagens 
import cv2 # Para manipulação de imagens
from PIL import Image,ImageEnhance # Para manipulação de imagens  
import numpy as np # Para manipulação de arrays 

# AI Pkgs # Para detecção de COVID-19 em imagens de raio-X do tórax 
import tensorflow as tf # Para manipulação de redes neurais  

def main(): # Função principal da aplicação web 
	"""Ferramenta simples para detecção de Covid-19 por radiografia de tórax""" 
	html_templ = """  
	<div style="background-color:blue;padding:10px;"> 
	<h1 style="color:yellow">Ferramenta de detecção de Covid-19</h1>
	</div> 
	""" 

	st.markdown(html_templ,unsafe_allow_html=True)
	st.write("Uma proposta simples para o diagnóstico de Covid-19 com tecnologia Deep Learning e Streamlit") 

	st.sidebar.image("covid19.jpeg",width=300) 

	image_file = st.sidebar.file_uploader("Carregue uma imagem de raio-X (jpg, png ou jpeg)",type=['jpg','png','jpeg']) 

	if image_file is not None: # Se o usuário carregar uma imagem de raio-X do tórax 
		our_image = Image.open(image_file) # Carregue a imagem de raio-X do tórax 

		if st.sidebar.button("Pré-visualização de imagem"): # Se o usuário clicar no botão "Pré-visualização de imagem" 
			st.sidebar.image(our_image,width=300) # Pré-visualização da imagem de raio-X do tórax 

		activities = ["Aprimoramento de imagem","Diagnóstico", "Isenção de responsabilidade e informações"] # Opções da interface web 
		choice = st.sidebar.selectbox("Selecione a atividade",activities) # Interface web 

		if choice == 'Image Enhancement': # Se o usuário selecionar a opção "Aprimoramento de imagem"
			st.subheader("Melhoria de imagem") # Interface web	

			enhance_type = st.sidebar.radio("Melhorar Tipo",["Original","Contraste","Brilho"]) # Interface web	

			if enhance_type == 'Contrast': # Se o usuário selecionar a opção "Contraste" 
				c_rate = st.slider("Contraste",0.5,5.0) # Interface web  
				enhancer = ImageEnhance.Contrast(our_image) # Interface web
				img_output = enhancer.enhance(c_rate) # Interface web
				st.image(img_output,use_column_width=True) # Interface web


			elif enhance_type == 'Brightness': 	# Se o usuário selecionar a opção "Brilho"
				c_rate = st.slider("Brilho",0.5,5.0) # Interface web 
				enhancer = ImageEnhance.Brightness(our_image) 	# Interface web
				img_output = enhancer.enhance(c_rate) # Interface web
				st.image(img_output,width=600,use_column_width=True) # Interface web


			else:
				st.text("Imagem original") # Interface web 
				st.image(our_image,width=600,use_column_width=True) # Interface web


		elif choice == 'Diagnosis': # Se o usuário selecionar a opção "Diagnóstico"
			
			if st.sidebar.button("Diagnóstico"): # Se o usuário clicar no botão "Diagnóstico" 

				# Image to Black and White # Converte a imagem de raio-X do tórax para preto e branco 
				new_img = np.array(our_image.convert('RGB')) # Converte a imagem de raio-X do tórax para RGB # nossa imagem é binária, temos que convertê-la em array 
				new_img = cv2.cvtColor(new_img,1) # 0 é original, 1 é escala de cinza
				gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY) # converte em escala de cinza 
				st.text("Raio-x do tórax") #
				st.image(gray,use_column_width=True)

				# PX-Ray (Image) Preprocessing # Pré-processamento da imagem de raio-X do tórax 
				IMG_SIZE = (200,200) # Tamanho da imagem de raio-X do tórax
				img = cv2.equalizeHist(gray) # Equalização do histograma da imagem de raio-X do tórax
				img = cv2.resize(img,IMG_SIZE) # Redimensionamento da imagem de raio-X do tórax
				img = img/255.  # Normalização da imagem de raio-X do tórax

				# Image reshaping according to Tensorflow format # Redimensionamento da imagem de raio-X do tórax de acordo com o formato do Tensorflow
				X_Ray = img.reshape(1,200,200,1) # Redimensionamento da imagem de raio-X do tórax

				# Pre-Trained CNN Model Importing # Importação do modelo CNN pré-treinado 
				model = tf.keras.models.load_model("./models/Covid19_CNN_Classifier.h5") # Importação do modelo CNN pré-treinado

				# Diagnosis (Prevision=Binary Classification) # Diagnóstico (Previsão = Classificação binária)
				diagnosis = model.predict_classes(X_Ray) # Diagnóstico (Previsão = Classificação binária)
				diagnosis_proba = model.predict(X_Ray) # Probabilidade do diagnóstico (Previsão = Classificação binária)
				probability_cov = diagnosis_proba*100 # Probabilidade do diagnóstico (Previsão = Classificação binária)
				probability_no_cov = (1-diagnosis_proba)*100 # Probabilidade do diagnóstico (Previsão = Classificação binária)

				my_bar = st.sidebar.progress(0) # Interface web


				for percent_complete in range(100): # para cada porcentagem completa
					time.sleep(0.05) # tempo de espera 
					my_bar.progress(percent_complete + 1) # progresso da barra de progresso

				# Diagnosis Cases: No-Covid=0, Covid=1 # Casos de diagnóstico: No-Covid = 0, Covid = 1
				if diagnosis == 0: # Se o diagnóstico for 0 (sem Covid-19)  
					st.sidebar.success("DIAGNÓSTICO: SEM COVID-19 (Probabilidade: %.2f%%)" % (probability_no_cov)) # sucesso na interface web
				else: # Se o diagnóstico for 1 (com Covid-19) 
					st.sidebar.error("DIAGNÓSTICO: COVID-19 (Probabilidade: %.2f%%)" % (probability_cov)) # sucesso na interface web 

				st.warning("Este Web App é apenas uma DEMO sobre Redes Neurais Artificiais, portanto não há valor clínico em seu diagnóstico e o autor não é médico!") # Aviso na interface web 


		else: # Se o usuário selecionar a opção "Sobre" 
			st.subheader("Disclaimer and Info") # Interface web
			st.subheader("Disclaimer") # Interface web 
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


	if st.sidebar.button("About the Author"): # Se o usuário selecionar a opção "Sobre o autor" 
		st.sidebar.subheader("Ferramenta de Detecção Covid19") # ferramenta de detecção Covid19
		st.sidebar.markdown("by [Ing. Rosario Moscato](https://www.youtube.com/channel/UCDn-FahQNJQOekLrOcR7-7Q)") # autor da ferramenta de detecção Covid19
		st.sidebar.markdown("[rosario.moscato@outlook.com](mailto:rosario.moscato@outlook.com)") # email do autor da ferramenta de detecção Covid19
		st.sidebar.text("All Rights Reserved (2020)") # Todos os direitos reservados (2022)


if __name__ == '__main__': # Se o nome do módulo for igual a main  
		main()	# Chama a função main() 

		# Fim do código