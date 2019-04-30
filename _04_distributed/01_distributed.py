# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import tensorflow_datasets as tfds

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

datasets,info = tfds.load(name='mnist',with_info=True,as_supervised=True)
mnist_train,mnist_test = datasets['train'],datasets['test']
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000; BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA*strategy.num_replicas_in_sync

def scale(image, label): return tf.cast(image,tf.float32)/255, label
train_dataset = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

with strategy.scope():
	model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(64,activation='relu'),
		tf.keras.layers.Dense(10,activation='softmax')])
	model.compile(loss='sparse_categorical_crossentropy',
		optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt_{epoch}')

def decay(epoch):
	if epoch < 3: return 1e-3
	if epoch < 7: return 1e-4
	return 1e-5

class PrintLR(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		print('\nLearning rate for epoch {} is {}'.format(
			epoch+1,model.optimizer.lr.numpy()))

callbacks = [
	tf.keras.callbacks.TensorBoard(log_dir='./logs'),
	tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True),
	tf.keras.callbacks.LearningRateScheduler(decay),
	PrintLR()]

path = 'saved_model/'
model.fit(train_dataset,epochs=10,callbacks=callbacks)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
eval_loss,eval_acc = model.evaluate(eval_dataset)
tf.keras.experimental.export_saved_model(model,path)

unreplicated_model = tf.keras.experimental.load_from_saved_model(path)
unreplicated_model.compile(loss='sparse_categorical_crossentropy',
	optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
eval_loss,eval_acc = unreplicated_model.evaluate(eval_dataset)
print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss,eval_acc))

with strategy.scope():
	replicated_model = tf.keras.experimental.load_from_saved_model(path)
	replicated_model.compile(loss='sparse_categorical_crossentropy',
		optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
	eval_loss,eval_acc = replicated_model.evaluate(eval_dataset)
	print ('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss,eval_acc))
