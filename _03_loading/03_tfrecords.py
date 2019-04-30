# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# tf.Example is a {"string": tf.train.Feature} mapping

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value,type(tf.constant(0))):
		value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

n_observations = int(1e4)
feature0 = np.random.choice([False,True],n_observations)
feature1 = np.random.randint(0,5,n_observations)
strings = np.array([b'cat',b'dog',b'chicken',b'horse',b'goat'])
feature2 = strings[feature1]
feature3 = np.random.randn(n_observations)

def serialize_example(feature0, feature1, feature2, feature3):
	feature = {
		'feature0':_int64_feature(feature0),
		'feature1':_int64_feature(feature1),
		'feature2':_bytes_feature(feature2),
		'feature3':_float_feature(feature3)}
	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()

# TFRecord files using tf.data

features_dataset = tf.data.Dataset.from_tensor_slices((feature0,feature1,feature2,feature3))
def tf_serialize_example(f0, f1, f2, f3):
	tf_string = tf.py_function(serialize_example,(f0,f1,f2,f3),tf.string)
	return tf.reshape(tf_string,())
serialized_features_dataset = features_dataset.map(tf_serialize_example)
def generator():
	for features in features_dataset:
		yield serialize_example(*features)
serialized_features_dataset = tf.data.Dataset.from_generator(generator,
	output_types=tf.string,output_shapes=())
writer = tf.data.experimental.TFRecordWriter('test.tfrecord')
writer.write(serialized_features_dataset)
feature_description = {
	'feature0':tf.io.FixedLenFeature([],tf.int64,default_value=0),
	'feature1':tf.io.FixedLenFeature([],tf.int64,default_value=0),
	'feature2':tf.io.FixedLenFeature([],tf.string,default_value=''),
	'feature3':tf.io.FixedLenFeature([],tf.float32,default_value=0.0)}
def _parse_function(example_proto):
	return tf.io.parse_single_example(example_proto,feature_description)
raw_dataset = tf.data.TFRecordDataset(['test.tfrecord'])
parsed_dataset = raw_dataset.map(_parse_function)

# TFRecord files in python

with tf.io.TFRecordWriter('test.tfrecord') as writer:
	for i in range(n_observations):
		example = serialize_example(feature0[i],feature1[i],feature2[i],feature3[i])
		writer.write(example)
raw_dataset = tf.data.TFRecordDataset(['test.tfrecord'])
for raw_record in raw_dataset.take(1):
	example = tf.train.Example()
	example.ParseFromString(raw_record.numpy())
	# print(example)

# Reading/Writing Image Data

cat_in_snow = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg',
	'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
	'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')
image_labels = {cat_in_snow:0,williamsburg_bridge:1}
image_string = open(cat_in_snow,'rb').read()
label = image_labels[cat_in_snow]
def image_example(image_string, label):
	image_shape = tf.image.decode_jpeg(image_string).shape
	feature = {
		'height':_int64_feature(image_shape[0]),
		'width':_int64_feature(image_shape[1]),
		'depth':_int64_feature(image_shape[2]),
		'label':_int64_feature(label),
		'image_raw':_bytes_feature(image_string)}
	return tf.train.Example(features=tf.train.Features(feature=feature))
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
	for filename,label in image_labels.items():
		image_string = open(filename,'rb').read()
		tf_example = image_example(image_string,label)
		writer.write(tf_example.SerializeToString())
raw_image_dataset = tf.data.TFRecordDataset(record_file)

image_feature_description = {
	'height':tf.io.FixedLenFeature([], tf.int64),
	'width':tf.io.FixedLenFeature([], tf.int64),
	'depth':tf.io.FixedLenFeature([], tf.int64),
	'label':tf.io.FixedLenFeature([], tf.int64),
	'image_raw': tf.io.FixedLenFeature([], tf.string)}
def _parse_image_function(example_proto):
	return tf.io.parse_single_example(example_proto,image_feature_description)
parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
for image_features in parsed_image_dataset:
	print(image_features['image_raw'].numpy())
# for raw_record in raw_image_dataset.take(1):
# 	example = tf.train.Example()
# 	print(example)
# 	print(example.ParseFromString(raw_record.numpy()))
