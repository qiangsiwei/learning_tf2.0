# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np, tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds

def rnn():
	dataset,info = tfds.load('imdb_reviews/subwords8k', with_info=True,as_supervised=True)
	data_tr,data_te = dataset['train'],dataset['test']
	tokenizer = info.features['text'].encoder
	data_tr = data_tr.shuffle(10000).padded_batch(64,data_tr.output_shapes)
	data_te = data_te.padded_batch(64,data_te.output_shapes)
	model = tf.keras.Sequential([
		tf.keras.layers.Embedding(tokenizer.vocab_size,64),
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
		tf.keras.layers.Dense(64,activation='relu'),
		tf.keras.layers.Dense(1,activation='sigmoid')])
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.fit(data_tr,epochs=10,validation_data=data_te)
	test_loss,test_acc = model.evaluate(data_te)
	print('Test Loss: {}'.format(test_loss))
	print('Test Accuracy: {}'.format(test_acc))
	def sample_predict(sentence, pad):
		def pad_to_size(vec, size):
			vec.extend([0]*(size-len(vec)))
		sentence = tokenizer.encode(sentence)
		if pad: sentence = pad_to_size(sentence,64)
		return model.predict(tf.expand_dims(sentence,0))
	sentence = ('The movie was cool. The animation and the graphics '
				'were out of this world. I would recommend this movie.')
	print(sample_predict(sentence,pad=False))

if __name__ == '__main__':
	rnn()
