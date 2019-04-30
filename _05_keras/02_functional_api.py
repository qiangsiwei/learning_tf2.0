# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,),name='img')
x = layers.Dense(64,activation='relu')(inputs)
x = layers.Dense(64,activation='relu')(x)
outputs = layers.Dense(10,activation='softmax')(x)
model = keras.Model(inputs=inputs,outputs=outputs,name='mnist_model')
keras.utils.plot_model(model,'my_first_model.png',show_shapes=True)
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255
model.compile(loss='sparse_categorical_crossentropy',
			  optimizer=keras.optimizers.RMSprop(),
			  metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=64,epochs=5,validation_split=0.2)
test_scores = model.evaluate(x_test,y_test,verbose=0)
print(test_scores)
model.save('path_to_my_model.h5'); del model
model = keras.models.load_model('path_to_my_model.h5')

# Using the same graph of layers to define multiple models

encoder_input = keras.Input(shape=(28,28,1),name='img')
x = layers.Conv2D(16,3,activation='relu')(encoder_input)
x = layers.Conv2D(32,3,activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32,3,activation='relu')(x)
x = layers.Conv2D(16,3,activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)
encoder = keras.Model(encoder_input,encoder_output,name='encoder')
x = layers.Reshape((4,4,1))(encoder_output)
x = layers.Conv2DTranspose(16,3,activation='relu')(x)
x = layers.Conv2DTranspose(32,3,activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16,3,activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1,3,activation='relu')(x)
autoencoder = keras.Model(encoder_input,decoder_output,name='autoencoder')

# All models are callable, just like layers

encoder_input = keras.Input(shape=(28,28,1),name='original_img')
x = layers.Conv2D(16,3,activation='relu')(encoder_input)
x = layers.Conv2D(32,3,activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32,3,activation='relu')(x)
x = layers.Conv2D(16,3,activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)
encoder = keras.Model(encoder_input,encoder_output,name='encoder')
decoder_input = keras.Input(shape=(16,),name='encoded_img')
x = layers.Reshape((4,4,1))(decoder_input)
x = layers.Conv2DTranspose(16,3,activation='relu')(x)
x = layers.Conv2DTranspose(32,3,activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16,3,activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1,3,activation='relu')(x)
decoder = keras.Model(decoder_input,decoder_output,name='decoder')
autoencoder_input = keras.Input(shape=(28,28,1),name='img')
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input,decoded_img,name='autoencoder')

def get_model():
	inputs = keras.Input(shape=(128,))
	outputs = layers.Dense(1,activation='sigmoid')(inputs)
	return keras.Model(inputs,outputs)
model1 = get_model()
model2 = get_model()
model3 = get_model()
inputs = keras.Input(shape=(128,))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = layers.average([y1,y2,y3])
ensemble_model = keras.Model(inputs=inputs,outputs=outputs)

# Manipulating complex graph topologies

num_tags = 12; num_words = 10000; num_departments = 4
title_input = keras.Input(shape=(None,),name='title')
body_input = keras.Input(shape=(None,),name='body')
tags_input = keras.Input(shape=(num_tags,),name='tags')
title_features = layers.Embedding(num_words,64)(title_input)
body_features = layers.Embedding(num_words,64)(body_input)
title_features = layers.LSTM(128)(title_features)
body_features = layers.LSTM(32)(body_features)
x = layers.concatenate([title_features,body_features,tags_input])
priority_pred = layers.Dense(1,activation='sigmoid',name='priority')(x)
department_pred = layers.Dense(num_departments,activation='softmax',name='department')(x)
model = keras.Model(inputs=[title_input,body_input,tags_input],outputs=[priority_pred,department_pred])
keras.utils.plot_model(model,'multi_input_and_output_model.png',show_shapes=True)
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
			  loss={'priority':'binary_crossentropy',
					'department':'categorical_crossentropy'},
			  loss_weights=[1.,0.2])
title_data = np.random.randint(num_words,size=(1280,10))
body_data = np.random.randint(num_words,size=(1280,100))
tags_data = np.random.randint(2,size=(1280,num_tags)).astype('float32')
priority_targets = np.random.random(size=(1280,1))
dept_targets = np.random.randint(2,size=(1280,num_departments))
model.fit({'title':title_data,'body':body_data,'tags':tags_data},
		  {'priority':priority_targets,'department':dept_targets},
		  epochs=2,batch_size=32)

# A toy resnet model (non-linear connectivity topologies)

inputs = keras.Input(shape=(32,32,3),name='img')
x = layers.Conv2D(32,3,activation='relu')(inputs)
x = layers.Conv2D(64,3,activation='relu')(x)
block_1_output = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(64,3,activation='relu',padding='same')(block_1_output)
x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
block_2_output = layers.add([x,block_1_output])
x = layers.Conv2D(64,3,activation='relu',padding='same')(block_2_output)
x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
block_3_output = layers.add([x,block_2_output])
x = layers.Conv2D(64,3,activation='relu')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256,activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10,activation='softmax')(x)
model = keras.Model(inputs,outputs,name='toy_resnet')
keras.utils.plot_model(model,'mini_resnet.png',show_shapes=True)
(x_train,y_train),(x_test,y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
			  loss='categorical_crossentropy',
			  metrics=['acc'])
model.fit(x_train,y_train,batch_size=64,epochs=1,validation_split=0.2)

# Sharing layers

shared_embedding = layers.Embedding(1000,128)
text_input_a = keras.Input(shape=(None,),dtype='int32')
text_input_b = keras.Input(shape=(None,),dtype='int32')
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)

# Extracting and reusing nodes in the graph of layers

from tensorflow.keras.applications import VGG19
vgg19 = VGG19()
features_list = [layer.output for layer in vgg19.layers]
feat_extraction_model = keras.Model(inputs=vgg19.input,outputs=features_list)
img = np.random.random((1,224,224,3)).astype('float32')
extracted_features = feat_extraction_model(img)

# Extending the API by writing custom layers

class CustomDense(layers.Layer):
	def __init__(self, units=32):
		super(CustomDense,self).__init__()
		self.units = units
	def build(self, input_shape):
		self.w = self.add_weight(shape=(input_shape[-1],self.units),initializer='random_normal',trainable=True)
		self.b = self.add_weight(shape=(self.units,),initializer='random_normal',trainable=True)
	def call(self, inputs):
		return tf.matmul(inputs,self.w)+self.b
	def get_config(self):
		return {'units':self.units}
	@classmethod
	def from_config(cls, config):
		return cls(**config)
inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)
model = keras.Model(inputs,outputs)
config = model.get_config()
new_model = keras.Model.from_config(config,custom_objects={'CustomDense':CustomDense})

# When to use the Functional API
# strengths:
#  - It is less verbose
#  - It validates your model while you're defining it
#  - Your Functional model is plottable and inspectable
#  - Your Functional model can be serialized or cloned
# weaknesses:
#  - It does not support dynamic architectures
#  - Sometimes, you just need to write everything from scratch
