# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
print(tf.__version__)

def classify_image():
	fashion_mnist = keras.datasets.fashion_mnist
	(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
	train_images,test_images = train_images/255.0,test_images/255.0
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28,28)),
		keras.layers.Dense(128,activation='relu'),
		keras.layers.Dense(10,activation='softmax')])
	model.summary()
	model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
	model.fit(train_images,train_labels,epochs=5)
	test_loss,test_acc = model.evaluate(test_images,test_labels)
	print(test_acc)
	predictions = model.predict(test_images)
	print(np.argmax(predictions[0]))

def classify_text():
	imdb = keras.datasets.imdb
	(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
	word_index = imdb.get_word_index()
	word_index = {k:(v+3) for k,v in word_index.items()}
	word_index.update({'<PAD>':0,'<START>':1,'<UNK>':2,'<UNUSED>':3})
	reverse_word_index = {value:key for key,value in word_index.items()}
	train_data = keras.preprocessing.sequence.pad_sequences(
		train_data,value=word_index['<PAD>'],padding='post',maxlen=256)
	test_data = keras.preprocessing.sequence.pad_sequences(
		test_data,value=word_index['<PAD>'],padding='post',maxlen=256)
	vocab_size = 10000
	model = keras.Sequential()
	model.add(keras.layers.Embedding(vocab_size,16))
	model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Dense(16,activation='relu'))
	model.add(keras.layers.Dense(1,activation='sigmoid'))
	model.summary()
	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
	x_tr = train_data[10000:]; y_tr = train_labels[10000:]
	x_te = train_data[:10000]; y_te = train_labels[:10000]
	history = model.fit(x_tr,y_tr,epochs=10,batch_size=512,validation_data=(x_te,y_te),verbose=1)
	results = model.evaluate(test_data,test_labels)
	print(results)

def classify_data(batch_size=5):
	from tensorflow import feature_column
	from tensorflow.keras import layers
	from sklearn.model_selection import train_test_split
	URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
	dataframe = pd.read_csv(URL)
	tr,te = train_test_split(dataframe,test_size=0.2)
	tr,va = train_test_split(tr,test_size=0.2)
	print(len(tr),len(va),len(te))
	def df_to_dataset(dataframe, shuffle=True, batch_size=32):
		dataframe,labels = dataframe.copy(),dataframe.pop('target')
		ds = tf.data.Dataset.from_tensor_slices((dict(dataframe),labels))
		if shuffle: ds = ds.shuffle(buffer_size=len(dataframe)).batch(batch_size)
		return ds
	tr_ds = df_to_dataset(tr,batch_size=batch_size)
	va_ds = df_to_dataset(va,shuffle=False,batch_size=batch_size)
	te_ds = df_to_dataset(te,shuffle=False,batch_size=batch_size)
	feature_columns = []
	for header in ['age','trestbps','chol','thalach','oldpeak','slope','ca']:
		feature_columns.append(feature_column.numeric_column(header))
	age = feature_column.numeric_column('age')
	age_buckets = feature_column.bucketized_column(age,boundaries=[18,25,30,35,40,45,50,55,60,65])
	feature_columns.append(age_buckets)
	thal = feature_column.categorical_column_with_vocabulary_list('thal',['fixed','normal','reversible'])
	feature_columns.append(feature_column.indicator_column(thal))
	feature_columns.append(feature_column.embedding_column(thal,dimension=8))
	crossed_feature = feature_column.crossed_column([age_buckets,thal],hash_bucket_size=1000)
	feature_columns.append(feature_column.indicator_column(crossed_feature))
	feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
	model = tf.keras.Sequential([
		feature_layer,
		layers.Dense(128,activation='relu'),
		layers.Dense(128,activation='relu'),
		layers.Dense(1,activation='sigmoid')])
	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
	model.fit(tr_ds,validation_data=va_ds,epochs=5)
	loss,accuracy = model.evaluate(te_ds); 
	print(accuracy)

def regression():
	dataset_path = keras.utils.get_file('auto-mpg.data','http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')
	column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
	raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values='?',comment='\t',sep=' ',skipinitialspace=True)
	dataset = raw_dataset.copy().dropna()
	origin = dataset.pop('Origin')
	dataset['USA'] = (origin==1)*1.0
	dataset['Eur'] = (origin==2)*1.0
	dataset['Jap'] = (origin==3)*1.0
	data_tr = dataset.sample(frac=0.8,random_state=0)
	data_te = dataset.drop(data_tr.index)
	labels_tr = data_tr.pop('MPG')
	labels_te = data_te.pop('MPG')
	print(data_tr.describe())
	train_stats = data_tr.describe().transpose()
	def norm(x):
		return (x-train_stats['mean'])/train_stats['std']
	normed_data_tr = norm(data_tr)
	normed_data_te = norm(data_te)
	def build_model():
		model = keras.Sequential([
			keras.layers.Dense(64,activation='relu',input_shape=[len(data_tr.keys())]),
			keras.layers.Dense(64,activation='relu'),
			keras.layers.Dense(1)])
		optimizer = tf.keras.optimizers.RMSprop(0.001)
		model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse'])
		return model
	model = build_model()
	model.summary()
	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
	history = model.fit(normed_data_tr,labels_tr,epochs=1000,
		validation_split=0.2,verbose=0,callbacks=[early_stop])
	loss,mae,mse = model.evaluate(normed_data_te,labels_te,verbose=0)
	print('Testing set Mean Abs Error: {:5.2f} MPG'.format(mae))

if __name__ == '__main__':
	# classify_image()
	# classify_text()
	# classify_data()
	regression()
