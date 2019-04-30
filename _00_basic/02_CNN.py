# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np, tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds

def cnn():
	(images_tr,labels_tr),(images_te,labels_te) = datasets.mnist.load_data()
	images_tr = images_tr.reshape((60000,28,28,1))
	images_te = images_te.reshape((10000,28,28,1))
	images_tr,images_te = images_tr/255.0,images_te/255.0
	model = models.Sequential()
	model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(64,(3,3),activation='relu'))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(64,(3,3),activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(64,activation='relu'))
	model.add(layers.Dense(10,activation='softmax'))
	model.summary()
	model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
	model.fit(images_tr,labels_tr,epochs=5)
	test_loss,test_acc = model.evaluate(images_te,labels_te)
	print(test_acc)

def transfer():
	initial_epochs = fine_tune_epochs = 10
	splits = tfds.Split.TRAIN.subsplit(weighted=(8,1,1))
	(raw_tr,raw_va,raw_te),metadata = tfds.load('cats_vs_dogs',split=list(splits),with_info=True,as_supervised=True)
	def format_example(image, label, img_size=160):
		return tf.image.resize((tf.cast(image,tf.float32)/127.5)-1,(img_size,img_size)), label
	data_tr = raw_tr.map(format_example).shuffle(1000).batch(32)
	data_va = raw_va.map(format_example).batch(32)
	data_te = raw_te.map(format_example).batch(32)
	base_model = tf.keras.applications.MobileNetV2(input_shape=(160,160,3),include_top=False,weights='imagenet')
	base_model.summary()
	model = tf.keras.Sequential([
		base_model,
		tf.keras.layers.GlobalAveragePooling2D(),
		keras.layers.Dense(1)])
	base_model.trainable = False
	model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])
	print(len(model.trainable_variables))
	model.fit(data_tr,epochs=initial_epochs,validation_data=data_va)
	base_model.trainable = True; fine_tune_at = 100
	for layer in base_model.layers[:fine_tune_at]:
		layer.trainable = False
	model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])
	print(len(model.trainable_variables))
	odel.fit(data_tr,epochs=initial_epochs+fine_tune_epochs,initial_epoch=initial_epochs,validation_data=data_va)

if __name__ == '__main__':
	cnn()
	transfer()
