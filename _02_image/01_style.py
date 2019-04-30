# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

con_path = tf.keras.utils.get_file('turtle.jpg',\
	'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
sty_path = tf.keras.utils.get_file('kandinsky.jpg',\
	'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

def load_img(path_to_img):
	img = tf.image.decode_jpeg(tf.io.read_file(path_to_img))
	img = tf.image.convert_image_dtype(img,tf.float32)
	shape = tf.cast(tf.shape(img)[:-1],tf.float32)
	new_shape = tf.cast(shape*512/max(shape),tf.int32)
	return tf.image.resize(img,new_shape)[tf.newaxis,:]

con_image = load_img(con_path)
sty_image = load_img(sty_path)

con_layers = ['block5_conv2']
sty_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

def vgg_layers(layer_names):
	vgg = tf.keras.applications.VGG19(include_top=False,weights='imagenet')
	vgg.trainable = False
	return tf.keras.Model([vgg.input],[vgg.get_layer(name).output for name in layer_names])

def gram_matrix(input_tensor):
	input_shape = tf.shape(input_tensor)
	return tf.linalg.einsum('bijc,bijd->bcd',input_tensor,input_tensor)/\
		tf.cast(input_shape[1]*input_shape[2],tf.float32)

class StyleContentModel(tf.keras.models.Model):
	def __init__(self, sty_layers, con_layers):
		super(StyleContentModel, self).__init__()
		self.vgg = vgg_layers(sty_layers+con_layers)
		self.sty_layers = sty_layers
		self.con_layers = con_layers
		self.n_sty_layers = len(sty_layers)
		self.vgg.trainable = False
	def call(self, input):
		preprocessed_input = tf.keras.applications.vgg19.preprocess_input(input*255.0)
		outputs = self.vgg(preprocessed_input)
		sty_outputs,con_outputs = (outputs[:self.n_sty_layers],outputs[self.n_sty_layers:])
		sty_outputs = [gram_matrix(sty_output) for sty_output in sty_outputs]
		sty_dict = {sty_name:value for sty_name,value in zip(self.sty_layers, sty_outputs)}
		con_dict = {con_name:value for con_name,value in zip(self.con_layers,con_outputs)}
		return {'sty':sty_dict,'con':con_dict}

extractor = StyleContentModel(sty_layers,con_layers)
sty_targets = extractor(sty_image)['sty']
con_targets = extractor(con_image)['con']
image = tf.Variable(con_image)
clip = lambda image:tf.clip_by_value(image,clip_value_min=0.0,clip_value_max=1.0)
opt = tf.optimizers.Adam(learning_rate=0.02,beta_1=0.99,epsilon=1e-1)

def sty_con_loss(outputs, sty_weight=1e-2, con_weight=1e4):
	sty_outputs = outputs['sty']
	con_outputs = outputs['con']
	sty_loss = tf.add_n([tf.reduce_mean((sty_outputs[name]-sty_targets[name])**2) for name in sty_outputs.keys()])
	sty_loss *= sty_weight/len(sty_layers)
	con_loss = tf.add_n([tf.reduce_mean((con_outputs[name]-con_targets[name])**2) for name in con_outputs.keys()])
	con_loss *= con_weight/len(con_layers)
	return sty_loss+con_loss

@tf.function()
def train_step(image):
	with tf.GradientTape() as tape:
		outputs = extractor(image)
		loss = sty_con_loss(outputs)
	grad = tape.gradient(loss,image)
	opt.apply_gradients([(grad,image)])
	image.assign(clip(image))

for n in tqdm(range(100)):
	train_step(image)

plt.imsave('output.png',image.read_value()[0])
