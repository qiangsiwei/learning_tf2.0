# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np, pandas as pd, tensorflow as tf

def boosted_trees():
	dftr = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
	dfev = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
	y_tr = dftr.pop('survived')
	y_ev = dfev.pop('survived')
	fc = tf.feature_column
	CAT_COLUMNS = ['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
	NUM_COLUMNS = ['age','fare']
	def one_hot_cat_column(feature_name, vocab):
		return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocab))
	feature_columns = []
	for feature_name in CAT_COLUMNS:
		feature_columns.append(one_hot_cat_column(feature_name,dftrain[feature_name].unique()))
	for feature_name in NUM_COLUMNS:
		feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))
	NUM_EXAMPLES = len(y_tr)
	def make_input_fn(X, y, n_epochs=None, shuffle=True):
		def input_fn():
			dataset = tf.data.Dataset.from_tensor_slices((dict(X),y))
			if shuffle: dataset = dataset.shuffle(NUM_EXAMPLES)
			dataset = dataset.repeat(n_epochs)
			dataset = dataset.batch(NUM_EXAMPLES)
			return dataset
		return input_fn
	tr_input_fn = make_input_fn(dftr,y_tr)
	ev_input_fn = make_input_fn(dfev,y_ev,n_epochs=1,shuffle=False)
	linear_est = tf.estimator.LinearClassifier(feature_columns)
	linear_est.train(tr_input_fn,max_steps=100)
	result = linear_est.evaluate(ev_input_fn)
	print(pd.Series(result))
	n_batches = 1
	est = tf.estimator.BoostedTreesClassifier(feature_columns,n_batches_per_layer=n_batches)
	est.train(tr_input_fn,max_steps=100)
	result = est.evaluate(ev_input_fn)
	print(pd.Series(result))

if __name__ == '__main__':
	boosted_trees()
