# -*- coding: utf-8 -*-

import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_model():
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(1,activation='linear',input_dim=784))
	model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.1),loss='mean_squared_error',metrics=['mae'])
	return model

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255

class MyCustomCallback(tf.keras.callbacks.Callback):
	def on_train_batch_begin(self, batch, logs=None):
		print('Training: batch {} begins at {}'.format(batch,datetime.datetime.now().time()))
	def on_train_batch_end(self, batch, logs=None):
		print('Training: batch {} ends at {}'.format(batch,datetime.datetime.now().time()))
	def on_test_batch_begin(self, batch, logs=None):
		print('Evaluating: batch {} begins at {}'.format(batch,datetime.datetime.now().time()))
	def on_test_batch_end(self, batch, logs=None):
		print('Evaluating: batch {} ends at {}'.format(batch,datetime.datetime.now().time()))

model = get_model()
_ = model.fit(x_train,y_train,batch_size=64,epochs=1,steps_per_epoch=5,verbose=0,callbacks=[MyCustomCallback()])
_ = model.evaluate(x_test,y_test,batch_size=128,verbose=0,steps=5,callbacks=[MyCustomCallback()])

# Model methods that take callbacks
#  - fit(), fit_generator()
#  - evaluate(), evaluate_generator()
#  - predict(), predict_generator()

# An overview of callback methods
#  - on_(train|test|predict)_begin(self, logs=None)
#  - on_(train|test|predict)_end(self, logs=None)
#  - on_(train|test|predict)_batch_begin(self, batch, logs=None)
#  - on_(train|test|predict)_batch_end(self, batch, logs=None)
# Training specific methods
#  - on_epoch_begin(self, epoch, logs=None)
#  - on_epoch_end(self, epoch, logs=None)

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
	def on_train_batch_end(self, batch, logs=None):
		print('For batch {}, loss is {:7.2f}.'.format(batch,logs['loss']))
	def on_test_batch_end(self, batch, logs=None):
		print('For batch {}, loss is {:7.2f}.'.format(batch,logs['loss']))
	def on_epoch_end(self, epoch, logs=None):
		print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch,logs['loss'],logs['mae']))
model = get_model()
_ = model.fit(x_train,y_train,batch_size=64,steps_per_epoch=5,epochs=3,verbose=0,callbacks=[LossAndErrorPrintingCallback()])
_ = model.evaluate(x_test,y_test,batch_size=128,verbose=0,steps=20,callbacks=[LossAndErrorPrintingCallback()])

# ----- Examples of Keras callback applications ----- 

# 1. Early stopping at minimum loss

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
	def __init__(self, patience=0):
		super(EarlyStoppingAtMinLoss,self).__init__()
		self.patience = patience
		self.best_weights = None
	def on_train_begin(self, logs=None):
		self.wait = 0
		self.stopped_epoch = 0
		self.best = np.Inf
	def on_epoch_end(self, epoch, logs=None):
		current = logs.get('loss')
		if np.less(current,self.best):
			self.best = current
			self.wait = 0
			self.best_weights = self.model.get_weights()
		else:
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				self.model.set_weights(self.best_weights)
	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0:
		  print('Epoch %05d: early stopping'%(self.stopped_epoch+1))
model = get_model()
_ = model.fit(x_train,y_train,batch_size=64,steps_per_epoch=5,epochs=30,verbose=0,
	callbacks=[LossAndErrorPrintingCallback(),EarlyStoppingAtMinLoss()])

# 2. Learning rate scheduling

class LearningRateScheduler(tf.keras.callbacks.Callback):
	def __init__(self, schedule):
		super(LearningRateScheduler,self).__init__()
		self.schedule = schedule
	def on_epoch_begin(self, epoch, logs=None):
		if not hasattr(self.model.optimizer,'lr'):
			raise ValueError('Optimizer must have a "lr" attribute.')
		lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
		scheduled_lr = self.schedule(epoch,lr)
		tf.keras.backend.set_value(self.model.optimizer.lr,scheduled_lr)
		print('\nEpoch %05d: Learning rate is %6.4f.'%(epoch,scheduled_lr))
LR_SCHEDULE = [(3,0.05),(6,0.01),(9,0.005),(12,0.001)]
def lr_schedule(epoch, lr):
	if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]: return lr
	for i in range(len(LR_SCHEDULE)):
		if epoch == LR_SCHEDULE[i][0]:
			return LR_SCHEDULE[i][1]
	return lr
model = get_model()
_ = model.fit(x_train,y_train,batch_size=64,steps_per_epoch=5,epochs=15,verbose=0,
	callbacks=[LossAndErrorPrintingCallback(), LearningRateScheduler(lr_schedule)])
