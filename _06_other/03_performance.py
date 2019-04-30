# -*- coding: utf-8 -*-

# ----- Input Pipeline Structure ----- 

#  - Extract: Read data from memory (NumPy) or persistent storage
#             either local (HDD or SSD) or remote (e.g. GCS or HDFS).
#  - Transform: Use CPU to parse and perform preprocessing operations on the data
#               such as shuffling, batching, and domain specific transformations
#               such as image decompression and augmentation, text vectorization, or video temporal sampling.
#  - Load: Load the transformed data onto the accelerator device(s) 
#          (e.g. GPU(s) or TPU(s)) that execute the machine learning model.

def parse_fn(example):
	example_fmt = {
		'image':tf.FixedLengthFeature((),tf.string,''),
		'label':tf.FixedLengthFeature((),tf.int64,-1)}
	parsed = tf.parse_single_example(example,example_fmt)
	image = tf.io.image.decode_image(parsed['image'])
	image = _augment_helper(image) # augments image using slice, reshape, resize_bilinear
	return image, parsed['label']

def make_dataset():
	dataset = tf.data.TFRecordDataset('/path/to/dataset/train-*.tfrecord')
	dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
	dataset = dataset.map(map_func=parse_fn)
	dataset = dataset.batch(batch_size=FLAGS.batch_size)
	return dataset

# ----- Optimizing Performance ----- 

# 1. Pipelining

# dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 2. Parallelize Data Transformation

# dataset = dataset.map(map_func=parse_fn,num_parallel_calls=tf.data.experimental.AUTOTUNE)

# 3. Parallelize Data Extraction
#  - Time-to-first-byte
#  - Read throughput

# files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
# dataset = files.interleave(tf.data.TFRecordDataset,
# 	cycle_length=FLAGS.num_parallel_reads,
#     num_parallel_calls=tf.data.experimental.AUTOTUNE)

# ----- Performance Considerations ----- 

# Map and Batch
# Map and Cache
# Map and Interleave / Prefetch / Shuffle
# Repeat and Shuffle

#  - repeat before shuffle -> better performance
#  - shuffle before repeat -> stronger ordering guarantees

# ----- Summary of Best Practices ----- 
