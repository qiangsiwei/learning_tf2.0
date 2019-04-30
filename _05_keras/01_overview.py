# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----- Sequential model -----

# - Configure the layers:
layers.Dense(64,activation='sigmoid')
layers.Dense(64,activation=tf.keras.activations.sigmoid)
layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l1(0.01))
layers.Dense(64,bias_regularizer=tf.keras.regularizers.l2(0.01))
layers.Dense(64,kernel_initializer='orthogonal')
layers.Dense(64,bias_initializer=tf.keras.initializers.Constant(2.0))

# !! Train and evaluate

model = tf.keras.Sequential([
	layers.Dense(64,activation='relu',input_shape=(32,)),
	layers.Dense(64,activation='relu'),
	layers.Dense(10,activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
# optimizer: tf.keras.optimizers or 'adam' or 'sgd'
# loss: tf.keras.losses or 'mse'
# metrics: tf.keras.metrics or 'accuracy'
# run_eagerly=True

data = np.random.random((1000,32))
labels = np.random.random((1000,10))
val_data = np.random.random((100,32))
val_labels = np.random.random((100,10))
model.fit(data,labels,epochs=10,batch_size=32,validation_data=(val_data,val_labels))
# epochs / batch_size / validation_data

# !! Train from tf.data datasets

dataset = tf.data.Dataset.from_tensor_slices((data,labels))
dataset = dataset.batch(32)
model.fit(dataset,epochs=10,steps_per_epoch=30)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data,val_labels))
val_dataset = val_dataset.batch(32)
model.fit(dataset,epochs=10,validation_data=val_dataset)

# Evaluate and predict

data = np.random.random((1000,32))
labels = np.random.random((1000,10))
model.evaluate(data,labels,batch_size=32)
model.evaluate(dataset,steps=30)
result = model.predict(data,batch_size=32)
print(result.shape)

# ----- Build advanced models -----

# - Multi-input models,
# - Multi-output models,
# - Models with shared layers (the same layer called several times),
# - Models with non-sequential data flows (e.g. residual connections).

# 1. A layer instance is callable and returns a tensor.
# 2. Input tensors and output tensors are used to define a tf.keras.Model instance.
# 3. This model is trained just like the Sequential model.

inputs = tf.keras.Input(shape=(32,))
x = layers.Dense(64,activation='relu')(inputs)
x = layers.Dense(64,activation='relu')(x)
predictions = layers.Dense(10,activation='softmax')(x)
model = tf.keras.Model(inputs=inputs,outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
model.fit(data,labels,batch_size=32,epochs=5)

# !! Model subclassing

# - Create layers in the __init__ method and set them as attributes of the class instance
# - Define the forward pass in the call method

class MyModel(tf.keras.Model):
	def __init__(self, num_classes=10):
		super(MyModel,self).__init__(name='my_model')
		self.num_classes = num_classes
		self.dense_1 = layers.Dense(32,activation='relu')
		self.dense_2 = layers.Dense(num_classes,activation='sigmoid')
	def call(self, inputs):
		x = self.dense_1(inputs)
		return self.dense_2(x)
model = MyModel(num_classes=10)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
model.fit(data,labels,batch_size=32, epochs=5)

# !! Custom layers

# __init__: Optionally define sublayers to be used by this layer.
# build: Create the weights of the layer. Add weights with the add_weight method.
# call: Define the forward pass.
# Optionally, a layer can be serialized by implementing the get_config method and the from_config class method.

class MyLayer(layers.Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(MyLayer,self).__init__(**kwargs)
	def build(self, input_shape):
		self.kernel = self.add_weight(name='kernel',
			shape=(input_shape[1],self.output_dim),initializer='uniform',trainable=True)
	def call(self, inputs):
		return tf.matmul(inputs,self.kernel)
	def get_config(self):
		base_config = super(MyLayer,self).get_config()
		base_config['output_dim'] = self.output_dim
		return base_config
	@classmethod
	def from_config(cls, config):
		return cls(**config)
model = tf.keras.Sequential([MyLayer(10),layers.Activation('softmax')])
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
model.fit(data,labels,batch_size=32,epochs=5)

# !! Callbacks

# - tf.keras.callbacks.ModelCheckpoint: Save checkpoints of your model at regular intervals.
# - tf.keras.callbacks.LearningRateScheduler: Dynamically change the learning rate.
# - tf.keras.callbacks.EarlyStopping: Interrupt training when validation performance has stopped improving.
# - tf.keras.callbacks.TensorBoard: Monitor the model's behavior using TensorBoard.

callbacks = [
	tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss'),
	tf.keras.callbacks.TensorBoard(log_dir='./logs')]
model.fit(data,labels,batch_size=32,epochs=5,callbacks=callbacks,validation_data=(val_data,val_labels))

# !! Save and restore

model = tf.keras.Sequential([
	layers.Dense(64, activation='relu', input_shape=(32,)),
	layers.Dense(10, activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.save_weights('./weights/my_model')
model.load_weights('./weights/my_model')
json_string = model.to_json()
yaml_string = model.to_yaml()
fresh_model = tf.keras.models.model_from_yaml(yaml_string)

model = tf.keras.Sequential([
	layers.Dense(10,activation='softmax',input_shape=(32,)),
	layers.Dense(10,activation='softmax')])
model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
model.fit(data,labels,batch_size=32,epochs=5)
model.save('my_model.h5')
model = tf.keras.models.load_model('my_model.h5')

# ----- Eager execution -----

# ----- Distribution -----

