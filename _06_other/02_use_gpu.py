# -*- coding: utf-8 -*-

import tensorflow as tf

tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

a = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape=[2,3],name='a')
b = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape=[3,2],name='b')
c = tf.matmul(a,b)
print(c)

# Manual device placement

with tf.device('/cpu:0'):
	a = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape=[2,3],name='a')
	b = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape=[3,2],name='b')
c = tf.matmul(a,b)
print(c)

# Allowing GPU memory growth

# tf.config.gpu.set_per_process_memory_growth()
# tf.config.gpu.set_per_process_memory_fraction(0.4)

# Using a single GPU on a multi-GPU system

# with tf.device('/device:GPU:2'):
# 	a = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape=[2,3],name='a')
# 	b = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape=[3,2],name='b')
# c = tf.matmul(a,b)
# print(c)

# Using multiple GPUs (data parallelism)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
	inputs = tf.keras.layers.Input(shape=(1,))
	predictions = tf.keras.layers.Dense(1)(inputs)
	model = tf.keras.models.Model(inputs=inputs,outputs=predictions)
	model.compile(loss='mse',optimizer=tf.keras.optimizers.RMSprop())

# !! Without tf.distribute.Strategy

# c = []
# for d in ['/device:GPU:2','/device:GPU:3']:
# 	with tf.device(d):
# 		a = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape=[2,3],name='a')
# 		b = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape=[3,2],name='b')
# 		c.append(tf.matmul(a,b))
# with tf.device('/cpu:0'):
# 	sum = tf.add_n(c)
# print(sum)
