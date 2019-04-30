# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
train_images = train_images[...,None]/np.float32(255)
test_images = test_images[...,None]/np.float32(255)

BUFFER_SIZE = len(train_images)
BATCH_SIZE_PER_REPLICA = 64; EPOCHS = 10
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*strategy.num_replicas_in_sync
train_steps_per_epoch = len(train_images)//GLOBAL_BATCH_SIZE
test_steps_per_epoch = len(test_images)//GLOBAL_BATCH_SIZE

with strategy.scope():
	train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
	train_iterator = strategy.make_dataset_iterator(train_dataset)
	test_dataset = tf.data.Dataset.from_tensor_slices((test_images,test_labels)).batch(GLOBAL_BATCH_SIZE) 
	test_iterator = strategy.make_dataset_iterator(test_dataset)

def create_model():
	model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(32,3,activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Conv2D(64,3,activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(64,activation='relu'),
		tf.keras.layers.Dense(10,activation='softmax')])
	return model

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')

with strategy.scope():
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	test_loss = tf.keras.metrics.Mean(name='test_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
	model = create_model()
	optimizer = tf.keras.optimizers.Adam()
	checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
	def train_step(inputs):
		images,labels = inputs
		with tf.GradientTape() as tape:
			predictions = model(images,training=True)
			loss = loss_object(labels,predictions)
		gradients = tape.gradient(loss,model.trainable_variables)
		optimizer.apply_gradients(zip(gradients,model.trainable_variables))
		train_loss(loss)
		train_accuracy(labels, predictions)
	def test_step(inputs):
		images,labels = inputs
		predictions = model(images,training=False)
		loss = loss_object(labels,predictions)
		test_loss(loss)
		test_accuracy(labels,predictions)
	@tf.function
	def distributed_train():
		return strategy.experimental_run(train_step,train_iterator)
	@tf.function
	def distributed_test():
		return strategy.experimental_run(test_step,test_iterator)
	for epoch in range(EPOCHS):
		train_iterator.initialize()
		for _ in range(train_steps_per_epoch):
			distributed_train()
		test_iterator.initialize()
		for _ in range(test_steps_per_epoch):
			distributed_test()
		if epoch%2 == 0:
			checkpoint.save(checkpoint_prefix)
		print(epoch+1,train_loss.result(),train_accuracy.result(),
					  test_loss.result(),test_accuracy.result())
		train_loss.reset_states()
		test_loss.reset_states()
		train_accuracy.reset_states()
		test_accuracy.reset_states()

eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')
new_model = create_model()
new_optimizer = tf.keras.optimizers.Adam()
test_dataset = tf.data.Dataset.from_tensor_slices((test_images,test_labels)).batch(GLOBAL_BATCH_SIZE)
@tf.function
def eval_step(images, labels):
	predictions = new_model(images,training=False)
	eval_accuracy(labels,predictions)
checkpoint = tf.train.Checkpoint(optimizer=new_optimizer,model=new_model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
for images,labels in test_dataset:
	eval_step(images,labels)
print(eval_accuracy.result())
