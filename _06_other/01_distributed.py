# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# Types of strategies
#  - Syncronous (all-reduce) vs asynchronous training
#  - Hardware platform

# ----- MirroredStrategy ----- 
# support synchronous distributed training on multiple GPUs on one machine

mirrored_strategy = tf.distribute.MirroredStrategy()
mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0','/gpu:1'])
mirrored_strategy = tf.distribute.MirroredStrategy(
	cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
# - tf.distribute.NcclAllReduce
# - tf.distribute.ReductionToOneDevice
# - tf.distribute.HierarchicalCopyAllReduce

# ----- MultiWorkerMirroredStrategy ----- 
# !! TF_CONFIG

multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    tf.distribute.experimental.CollectiveCommunication.NCCL)
# - CollectiveCommunication.RING/NCCL/AUTO

# ----- TPUStrategy ----- 

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)

# ----- ParameterServerStrategy ----- 
# !! TF_CONFIG

ps_strategy = tf.distribute.experimental.ParameterServerStrategy()

# ----- Using tf.distribute.Strategy with Keras ----- 

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
	model = tf.keras.Sequential([tf.keras.layers.Dense(1,input_shape=(1,))])
	model.compile(loss='mse',optimizer='sgd')
dataset = tf.data.Dataset.from_tensors(([1.],[1.])).repeat(100).batch(10)
model.fit(dataset,epochs=2)
model.evaluate(dataset)
inputs,targets = np.ones((100,1)),np.ones((100,1))
model.fit(inputs,targets,epochs=2,batch_size=10)
# BATCH_SIZE_PER_REPLICA = 5
# global_batch_size = (BATCH_SIZE_PER_REPLICA*mirrored_strategy.num_replicas_in_sync)
# dataset = tf.data.Dataset.from_tensors(([1.],[1.])).repeat(100)
# dataset = dataset.batch(global_batch_size)
# LEARNING_RATES_BY_BATCH_SIZE = {5:0.1,10:0.15}
# learning_rate = LEARNING_RATES_BY_BATCH_SIZE[global_batch_size]

# ----- Using tf.distribute.Strategy with Estimator ----- 

mirrored_strategy = tf.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(
	train_distribute=mirrored_strategy,eval_distribute=mirrored_strategy)
regressor = tf.estimator.LinearRegressor(
	feature_columns=[tf.feature_column.numeric_column('feats')],
    optimizer='SGD',
    config=config)
def input_fn():
	dataset = tf.data.Dataset.from_tensors(({'feats':[1.]},[1.]))
	return dataset.repeat(1000).batch(10)
regressor.train(input_fn=input_fn,steps=10)
regressor.evaluate(input_fn=input_fn,steps=10)

# ----- Using tf.distribute.Strategy with custom training loops ----- 

with mirrored_strategy.scope():
	model = tf.keras.Sequential([tf.keras.layers.Dense(1,input_shape=(1,))])
	optimizer = tf.keras.optimizers.SGD()
	dataset = tf.data.Dataset.from_tensors(([1.],[1.])).repeat(1000).batch(global_batch_size)
	input_iterator = mirrored_strategy.make_dataset_iterator(dataset)

@tf.function
def train_step():
	def step_fn(inputs):
		features,labels = inputs
		with tf.GradientTape() as tape:
			logits = model(features)
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
			loss = tf.reduce_sum(cross_entropy)*(1.0/global_batch_size)
		grads = tape.gradient(loss,model.trainable_variables)
		optimizer.apply_gradients(list(zip(grads,model.trainable_variables)))
		return loss
	per_replica_losses = mirrored_strategy.experimental_run(step_fn,input_iterator)
	mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,per_replica_losses)
	return mean_loss

with mirrored_strategy.scope():
	input_iterator.initialize()
	for _ in range(10):
		print(train_step())
