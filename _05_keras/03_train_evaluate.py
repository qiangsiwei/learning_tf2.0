# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----- Part I: Using build-in training & evaluation loops ----- 

inputs = keras.Input(shape=(784,),name='digits')
x = layers.Dense(64,activation='relu',name='dense_1')(inputs)
x = layers.Dense(64,activation='relu',name='dense_2')(x)
outputs = layers.Dense(10,activation='softmax',name='predictions')(x)
model = keras.Model(inputs=inputs,outputs=outputs)
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
model.compile(optimizer=keras.optimizers.RMSprop(),
			  loss=keras.losses.SparseCategoricalCrossentropy(),
			  metrics=[keras.metrics.SparseCategoricalAccuracy()])
history = model.fit(x_train,y_train,batch_size=64,epochs=3,validation_data=(x_val,y_val))
results = model.evaluate(x_test,y_test,batch_size=128)
predictions = model.predict(x_test[:3])

# Many built-in optimizers, losses, and metrics are available
#  - Optimizers: SGD()/RMSprop()/Adam()
#  - Losses: MeanSquaredError()/KLDivergence()/CosineSimilarity()
#  - Metrics: AUC()/Precision()/Recall()

# Writing custom losses and metrics

class CatgoricalTruePositives(keras.metrics.Metric):
	def __init__(self, name='binary_true_positives', **kwargs):
		super(CatgoricalTruePositives,self).__init__(name=name,**kwargs)
		self.true_positives = self.add_weight(name='tp',initializer='zeros')
	def update_state(self, y_true, y_pred, sample_weight=None):
		y_pred = tf.argmax(y_pred)
		values = tf.equal(tf.cast(y_true,'int32'),tf.cast(y_pred,'int32'))
		values = tf.cast(values,'float32')
		if sample_weight is not None:
			sample_weight = tf.cast(sample_weight,'float32')
			values = tf.multiply(values,sample_weight)
		return self.true_positives.assign_add(tf.reduce_sum(values))
	def result(self):
		return tf.identity(self.true_positives)
	def reset_states(self):
		self.true_positives.assign(0.)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
			  loss=keras.losses.SparseCategoricalCrossentropy(),
			  metrics=[CatgoricalTruePositives()])
model.fit(x_train,y_train,batch_size=64,epochs=3)

# Handling losses and metrics that don't fit the standard signature

class ActivityRegularizationLayer(layers.Layer):
	def call(self, inputs):
		self.add_loss(tf.reduce_sum(inputs)*0.1)
		return inputs
inputs = keras.Input(shape=(784,),name='digits')
x = layers.Dense(64,activation='relu',name='dense_1')(inputs)
x = ActivityRegularizationLayer()(x)
x = layers.Dense(64,activation='relu',name='dense_2')(x)
outputs = layers.Dense(10,activation='softmax',name='predictions')(x)
model = keras.Model(inputs=inputs,outputs=outputs)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
			  loss='sparse_categorical_crossentropy')
model.fit(x_train,y_train,batch_size=64,epochs=1)

class MetricLoggingLayer(layers.Layer):
	def call(self, inputs):
		self.add_metric(keras.backend.std(inputs),name='std_of_activation',aggregation='mean')
		return inputs
inputs = keras.Input(shape=(784,),name='digits')
x = layers.Dense(64,activation='relu',name='dense_1')(inputs)
x = MetricLoggingLayer()(x)
x = layers.Dense(64,activation='relu',name='dense_2')(x)
outputs = layers.Dense(10,activation='softmax',name='predictions')(x)
model = keras.Model(inputs=inputs,outputs=outputs)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
			  loss='sparse_categorical_crossentropy')
model.fit(x_train,y_train,batch_size=64,epochs=1)
inputs = keras.Input(shape=(784,),name='digits')
x1 = layers.Dense(64,activation='relu',name='dense_1')(inputs)
x2 = layers.Dense(64,activation='relu',name='dense_2')(x1)
outputs = layers.Dense(10,activation='softmax',name='predictions')(x2)
model = keras.Model(inputs=inputs,outputs=outputs)
model.add_loss(tf.reduce_sum(x1)*0.1)
model.add_metric(keras.backend.std(x1),name='std_of_activation',aggregation='mean')
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
			  loss='sparse_categorical_crossentropy')
model.fit(x_train,y_train,batch_size=64,epochs=1)

# Automatically setting apart a validation holdout set

def get_uncompiled_model():
	inputs = keras.Input(shape=(784,),name='digits')
	x = layers.Dense(64,activation='relu',name='dense_1')(inputs)
	x = layers.Dense(64,activation='relu',name='dense_2')(x)
	outputs = layers.Dense(10,activation='softmax',name='predictions')(x)
	model = keras.Model(inputs=inputs,outputs=outputs)
	return model
def get_compiled_model():
	model = get_uncompiled_model()
	model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
				  loss='sparse_categorical_crossentropy',
				  metrics=['sparse_categorical_accuracy'])
	return model

model = get_compiled_model()
model.fit(x_train,y_train,batch_size=64,validation_split=0.2,epochs=3)

model = get_compiled_model()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataset = test_dataset.batch(64)
model.fit(train_dataset,epochs=3)
model.evaluate(test_dataset)

model = get_compiled_model()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
model.fit(train_dataset,epochs=3,steps_per_epoch=100)

model = get_compiled_model()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_dataset = val_dataset.batch(64)
model.fit(train_dataset,epochs=3,validation_data=val_dataset,validation_steps=10)

# Other input formats supported
#  - Pandas dataframes or from Python generators

# Using sample weighting and class weighting

class_weight = {0:1.,1:1.,2:1.,3:1.,4:1.,5:2.,6:1.,7:1.,8:1.,9:1.}
model.fit(x_train,y_train,class_weight=class_weight,batch_size=64,epochs=4)
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train==5] = 2.
model = get_compiled_model()
model.fit(x_train,y_train,sample_weight=sample_weight,batch_size=64,epochs=4)
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train==5] = 2.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train,sample_weight))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
model = get_compiled_model()
model.fit(train_dataset,epochs=3)

# Passing data to multi-input, multi-output models

image_input = keras.Input(shape=(32,32,3),name='img_input')
timeseries_input = keras.Input(shape=(None,10),name='ts_input')
x1 = layers.Conv2D(3,3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)
x2 = layers.Conv1D(3,3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)
x = layers.concatenate([x1,x2])
score_output = layers.Dense(1,name='score_output')(x)
class_output = layers.Dense(5,activation='softmax',name='class_output')(x)
model = keras.Model(inputs=[image_input,timeseries_input],
					outputs=[score_output,class_output])
keras.utils.plot_model(model,'multi_input_and_output_model.png',show_shapes=True)
model.compile(
	optimizer=keras.optimizers.RMSprop(1e-3),
	loss={'score_output':keras.losses.MeanSquaredError(),
		  'class_output':keras.losses.CategoricalCrossentropy()},
	metrics={'score_output':[keras.metrics.MeanAbsolutePercentageError(),
							 keras.metrics.MeanAbsoluteError()],
			 'class_output':[keras.metrics.CategoricalAccuracy()]},
	loss_weight={'score_output':2.,'class_output':1.})
img_data = np.random.random_sample(size=(100,32,32,3))
ts_data = np.random.random_sample(size=(100,20,10))
score_targets = np.random.random_sample(size=(100,1))
class_targets = np.random.random_sample(size=(100,5))
# model.fit([img_data,ts_data],[score_targets,class_targets],batch_size=32,epochs=3)
model.fit({'img_input':img_data,'ts_input':ts_data},
		  {'score_output':score_targets,'class_output':class_targets},
		  batch_size=32,epochs=3)
train_dataset = tf.data.Dataset.from_tensor_slices((
	{'img_input':img_data,'ts_input':ts_data},
    {'score_output':score_targets,'class_output':class_targets}))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
model.fit(train_dataset,epochs=3)

# ----- Using callbacks -----

# Many built-in callbacks are available
#  - ModelCheckpoint
#  - EarlyStopping
#  - TensorBoard
#  - CSVLogger

model = get_compiled_model()
callbacks = [keras.callbacks.ModelCheckpoint(
	filepath='mymodel_{epoch}.h5',save_best_only=True,monitor='val_loss',verbose=1)]
model.fit(x_train,y_train,epochs=3,batch_size=64,callbacks=callbacks,validation_split=0.2)

model = get_compiled_model()
callbacks = [keras.callbacks.EarlyStopping(
	monitor='val_loss',min_delta=1e-2,patience=2,verbose=1)]
model.fit(x_train,y_train,epochs=20,batch_size=64,callbacks=callbacks,validation_split=0.2)

# Writing your own callback
class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs):
		self.losses = []
	def on_batch_end(self, batch, logs):
		self.losses.append(logs.get('loss'))

# Using learning rate schedules
# ExponentialDecay/PiecewiseConstantDecay/PolynomialDecay/InverseTimeDecay
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
	initial_learning_rate,decay_steps=100000,decay_rate=0.96,staircase=True)
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

# Visualizing loss and metrics during training
tensorboard_cbk = keras.callbacks.TensorBoard(log_dir='./logs')
model.fit(train_dataset,epochs=10,callbacks=[tensorboard_cbk])

# ----- Part II: Writing your own training & evaluation loops from scratch ----- 

inputs = keras.Input(shape=(784,),name='digits')
x = layers.Dense(64,activation='relu',name='dense_1')(inputs)
x = layers.Dense(64,activation='relu',name='dense_2')(x)
outputs = layers.Dense(10,activation='softmax',name='predictions')(x)
model = keras.Model(inputs=inputs,outputs=outputs)
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy()
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
for epoch in range(3):
	for step,(x_batch_train,y_batch_train) in enumerate(train_dataset):
		with tf.GradientTape() as tape:
			logits = model(x_batch_train)
			loss_value = loss_fn(y_batch_train,logits)
		grads = tape.gradient(loss_value,model.trainable_variables)
		optimizer.apply_gradients(zip(grads,model.trainable_variables))
		if (step+1)%200 == 0: print(step,float(loss_value))

# Low-level handling of metrics

inputs = keras.Input(shape=(784,),name='digits')
x = layers.Dense(64,activation='relu',name='dense_1')(inputs)
x = layers.Dense(64,activation='relu',name='dense_2')(x)
outputs = layers.Dense(10,activation='softmax',name='predictions')(x)
model = keras.Model(inputs=inputs,outputs=outputs)
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy()
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_dataset = val_dataset.batch(64)
for epoch in range(3):
	for step,(x_batch_train,y_batch_train) in enumerate(train_dataset):
		with tf.GradientTape() as tape:
			logits = model(x_batch_train)
			loss_value = loss_fn(y_batch_train,logits)
		grads = tape.gradient(loss_value,model.trainable_variables)
		optimizer.apply_gradients(zip(grads,model.trainable_variables))
		train_acc_metric(y_batch_train,logits)
		if (step+1)%200 == 0: print(step,float(loss_value))
	train_acc = train_acc_metric.result()
	print(float(train_acc))
	train_acc_metric.reset_states()
	for x_batch_val,y_batch_val in val_dataset:
		val_logits = model(x_batch_val)
		val_acc_metric(y_batch_val,val_logits)
	val_acc = val_acc_metric.result()
	print(float(val_acc))
	val_acc_metric.reset_states()
	
# Low-level handling of extra losses

class ActivityRegularizationLayer(layers.Layer):
	def call(self, inputs):
		self.add_loss(1e-2*tf.reduce_sum(inputs))
		return inputs
inputs = keras.Input(shape=(784,),name='digits')
x = layers.Dense(64,activation='relu',name='dense_1')(inputs)
x = ActivityRegularizationLayer()(x)
x = layers.Dense(64,activation='relu',name='dense_2')(x)
outputs = layers.Dense(10,activation='softmax',name='predictions')(x)
model = keras.Model(inputs=inputs,outputs=outputs)
logits = model(x_train)
logits = model(x_train[:64])
print(model.losses)
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
for epoch in range(3):
	for step,(x_batch_train,y_batch_train) in enumerate(train_dataset):
		with tf.GradientTape() as tape:
			logits = model(x_batch_train)
			loss_value = loss_fn(y_batch_train,logits)
			loss_value += sum(model.losses)
		grads = tape.gradient(loss_value,model.trainable_variables)
		optimizer.apply_gradients(zip(grads,model.trainable_variables))
		if (step+1)%200 == 0: print(step,float(loss_value))
		