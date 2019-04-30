# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,),name='digits')
x = layers.Dense(64,activation='relu',name='dense_1')(inputs)
x = layers.Dense(64,activation='relu',name='dense_2')(x)
outputs = layers.Dense(10,activation='softmax',name='predictions')(x)
model = keras.Model(inputs=inputs,outputs=outputs,name='3_layer_mlp')
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255
model.compile(loss='sparse_categorical_crossentropy',
			  optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train,y_train,batch_size=64,epochs=1)
predictions = model.predict(x_test)

# Whole-model saving
#  - The model's architecture
#  - The model's weight values
#  - The model's training config
#  - The optimizer and its state
model.save('path_to_my_model.h5')
new_model = keras.models.load_model('path_to_my_model.h5')
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions,new_predictions,atol=1e-6)

# Export to SavedModel (TensorFlow serving)
keras.experimental.export_saved_model(model,'path_to_saved_model')
new_model = keras.experimental.load_from_saved_model('path_to_saved_model')
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions,new_predictions,atol=1e-6)

# Architecture-only saving
config = model.get_config()
reinitialized_model = keras.Model.from_config(config)
new_predictions = reinitialized_model.predict(x_test)
assert abs(np.sum(predictions-new_predictions)) > 0.

json_config = model.to_json()
reinitialized_model = keras.models.model_from_json(json_config)

# Weights-only saving
weights = model.get_weights()
model.set_weights(weights)

config = model.get_config()
weights = model.get_weights()
new_model = keras.Model.from_config(config)
new_model.set_weights(weights)
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions,new_predictions,atol=1e-6)

json_config = model.to_json()
with open('model_config.json','w') as json_file:
    json_file.write(json_config)
model.save_weights('path_to_my_weights.h5')
with open('model_config.json') as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights('path_to_my_weights.h5')
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions,new_predictions,atol=1e-6)

model.save('path_to_my_model.h5'); del model
model = keras.models.load_model('path_to_my_model.h5')

# ----- Saving Subclassed Models ----- 

class ThreeLayerMLP(keras.Model):
	def __init__(self, name=None):
		super(ThreeLayerMLP,self).__init__(name=name)
		self.dense_1 = layers.Dense(64,activation='relu',name='dense_1')
		self.dense_2 = layers.Dense(64,activation='relu',name='dense_2')
		self.pred_layer = layers.Dense(10,activation='softmax',name='predictions')
	def call(self, inputs):
		x = self.dense_1(inputs)
		x = self.dense_2(x)
		return self.pred_layer(x)
def get_model():
	return ThreeLayerMLP(name='3_layer_mlp')
model = get_model()
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255
model.compile(loss='sparse_categorical_crossentropy',
			  optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train,y_train,batch_size=64,epochs=1)
model.save_weights('path_to_my_weights',save_format='tf')
predictions = model.predict(x_test)
first_batch_loss = model.train_on_batch(x_train[:64],y_train[:64])

# Note that in order to restore the optimizer state and the state of any stateful metric,
# you should compile the model (with the exact same arguments as before) and 
# call it on some data before calling load_weights

new_model = get_model()
new_model.compile(loss='sparse_categorical_crossentropy',
				  optimizer=keras.optimizers.RMSprop())
new_model.train_on_batch(x_train[:1],y_train[:1])
new_model.load_weights('path_to_my_weights')
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions,new_predictions,atol=1e-6)
new_first_batch_loss = new_model.train_on_batch(x_train[:64],y_train[:64])
assert first_batch_loss == new_first_batch_loss
