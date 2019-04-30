# -*- coding: utf-8 -*-

import os, time
import numpy as np
import tensorflow as tf

seq_length = 100
path_to_file = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file,'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
examples_per_epoch = len(text)//seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1,drop_remainder=True)
dataset = sequences.map(lambda chunk:(chunk[:-1],chunk[1:]))
dataset = dataset.shuffle(10000).batch(64,drop_remainder=True)
vocab_size = len(vocab); embedding_dim = 256; rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
	model = tf.keras.Sequential([
		tf.keras.layers.Embedding(vocab_size,embedding_dim,batch_input_shape=[batch_size,None]),
		tf.keras.layers.LSTM(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'),
		tf.keras.layers.Dense(vocab_size)])
	return model
model = build_model(vocab_size,embedding_dim,rnn_units,64)
model.summary()
def loss(labels, logits):
	return tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True)
model.compile(optimizer='adam', loss=loss)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True)
history = model.fit(dataset,epochs=10,callbacks=[checkpoint_callback])
model = build_model(vocab_size,embedding_dim,rnn_units,1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1,None]))
model.summary()
def generate_text(model, start_string):
	num_generate = 1000
	input_eval = [char2idx[s] for s in start_string]
	input_eval = tf.expand_dims(input_eval,0)
	text_generated = []
	temperature = 1.0
	model.reset_states()
	for i in range(num_generate):
		predictions = model(input_eval)
		predictions = tf.squeeze(predictions,0)
		predictions = predictions/temperature
		predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()
		input_eval = tf.expand_dims([predicted_id],0)
		text_generated.append(idx2char[predicted_id])
	return(start_string+''.join(text_generated))
print(generate_text(model,start_string=u'ROMEO: '))
model = build_model(vocab_size,embedding_dim,rnn_units,64)
optimizer = tf.keras.optimizers.Adam()
@tf.function
def train_step(inp, target):
	with tf.GradientTape() as tape:
		predictions = model(inp)
		loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(target,predictions))
	grads = tape.gradient(loss,model.trainable_variables)
	optimizer.apply_gradients(zip(grads,model.trainable_variables))
	return loss
for epoch in range(10):
	for batch_n,(inp,target) in enumerate(dataset):
		loss = train_step(inp, target)
		if (batch_n+1)%100 == 0:
			print(template.format(epoch+1,batch_n,loss))
	if (epoch+1)%5 == 0:
		model.save_weights(checkpoint_prefix.format(epoch=epoch))
model.save_weights(checkpoint_prefix.format(epoch=epoch))
