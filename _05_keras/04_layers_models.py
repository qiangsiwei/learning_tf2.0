# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers

# ----- The Layer class -----

class Linear(layers.Layer):
	def __init__(self, units=32, input_dim=32):
		super(Linear,self).__init__()
		w_init = tf.random_normal_initializer()
		self.w = tf.Variable(initial_value=w_init(shape=(input_dim,units),dtype='float32'),trainable=True)
		# self.w = self.add_weight(shape=(input_dim,units),initializer='random_normal',trainable=True)
		b_init = tf.zeros_initializer()
		self.b = tf.Variable(initial_value=b_init(shape=(units,),dtype='float32'),trainable=True)
		# self.b = self.add_weight(shape=(units,),initializer='zeros',trainable=True)
	def call(self, inputs):
		return tf.matmul(inputs, self.w) + self.b
x = tf.ones((2,2))
linear_layer = Linear(4,2)
y = linear_layer(x)
print(y)
assert linear_layer.weights == [linear_layer.w,linear_layer.b]

# Layers can have non-trainable weights

class ComputeSum(layers.Layer):
	def __init__(self, input_dim):
		super(ComputeSum,self).__init__()
		self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),trainable=False)
	def call(self, inputs):
		self.total.assign_add(tf.reduce_sum(inputs,axis=0))
		return self.total
x = tf.ones((2,2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())

# Best practice: deferring weight creation until the shape of the inputs is known

class Linear(layers.Layer):
	def __init__(self, units=32):
		super(Linear,self).__init__()
		self.units = units
	def build(self, input_shape):
		self.w = self.add_weight(shape=(input_shape[-1],self.units),initializer='random_normal',trainable=True)
		self.b = self.add_weight(shape=(self.units,),initializer='random_normal',trainable=True)
	def call(self, inputs):
		return tf.matmul(inputs,self.w)+self.b

# Layers are recursively composable

class MLPBlock(layers.Layer):
	def __init__(self):
		super(MLPBlock,self).__init__()
		self.linear_1 = Linear(32)
		self.linear_2 = Linear(32)
		self.linear_3 = Linear(1)
	def call(self, inputs):
		x = self.linear_1(inputs)
		x = tf.nn.relu(x)
		x = self.linear_2(x)
		x = tf.nn.relu(x)
		return self.linear_3(x)
mlp = MLPBlock()
y = mlp(tf.ones(shape=(3,64)))

# Layers recursively collect losses created during the forward pass

class ActivityRegularizationLayer(layers.Layer):
	def __init__(self, rate=1e-2):
		super(ActivityRegularizationLayer,self).__init__()
		self.rate = rate
	def call(self, inputs):
		self.add_loss(self.rate*tf.reduce_sum(inputs))
		return inputs
class OuterLayer(layers.Layer):
	def __init__(self):
		super(OuterLayer,self).__init__()
		self.activity_reg = ActivityRegularizationLayer(1e-2)
	def call(self, inputs):
		return self.activity_reg(inputs)
layer = OuterLayer()
assert len(layer.losses) == 0
_ = layer(tf.zeros(1,1))
assert len(layer.losses) == 1
_ = layer(tf.zeros(1,1))
assert len(layer.losses) == 1

class OuterLayer(layers.Layer):
	def __init__(self):
		super(OuterLayer,self).__init__()
		self.dense = layers.Dense(32,kernel_regularizer=tf.keras.regularizers.l2(1e-3))
	def call(self, inputs):
		return self.dense(inputs)
layer = OuterLayer()
_ = layer(tf.zeros((1,1)))
print(layer.losses)
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
# loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# for x_batch_train, y_batch_train in train_dataset:
# 	with tf.GradientTape() as tape:
# 		logits = layer(x_batch_train)
# 		loss_value = loss_fn(y_batch_train,logits))
# 		loss_value += sum(model.losses)
# 		grads = tape.gradient(loss_value,model.trainable_variables)
# 		optimizer.apply_gradients(zip(grads,model.trainable_variables))

# You can optionally enable serialization on your layers

# Privileged training argument in the call method

class CustomDropout(layers.Layer):
	def __init__(self, rate, **kwargs):
		super(CustomDropout,self).__init__(**kwargs)
		self.rate = rate
	def call(self, inputs, training=None):
		if training:
			return tf.nn.dropout(inputs,rate=self.rate)
		return inputs

# ----- Building Models -----

# The Model class has the same API as Layer, with the following differences:
#  - It exposes built-in training, evaluation, and prediction loops (model.fit(), model.evaluate(), model.predict()).
#  - It exposes the list of its inner layers, via the model.layers property.
#  - It exposes saving and serialization APIs.

class ResNet(tf.keras.Model):
	def __init__(self):
		super(ResNet, self).__init__()
		self.block_1 = ResNetBlock()
		self.block_2 = ResNetBlock()
		self.global_pool = layers.GlobalAveragePooling2D()
		self.classifier = Dense(num_classes)
	def call(self, inputs):
		x = self.block_1(inputs)
		x = self.block_2(x)
		x = self.global_pool(x)
		return self.classifier(x)
# resnet = ResNet()
# dataset = ...
# resnet.fit(dataset,epochs=10)
# resnet.save_weights(filepath)

# Putting it all together: an end-to-end example

class Sampling(layers.Layer):
	def call(self, inputs):
		z_mean,z_log_var = inputs
		batch = tf.shape(z_mean)[0]
		dim = tf.shape(z_mean)[1]
		epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
		return z_mean+tf.exp(0.5*z_log_var)*epsilon

class Encoder(layers.Layer):
	def __init__(self, latent_dim=32, intermediate_dim=64, name='encoder', **kwargs):
		super(Encoder,self).__init__(name=name,**kwargs)
		self.dense_proj = layers.Dense(intermediate_dim,activation='relu')
		self.dense_mean = layers.Dense(latent_dim)
		self.dense_log_var = layers.Dense(latent_dim)
		self.sampling = Sampling()
	def call(self, inputs):
		x = self.dense_proj(inputs)
		z_mean = self.dense_mean(x)
		z_log_var = self.dense_log_var(x)
		z = self.sampling((z_mean,z_log_var))
		return z_mean, z_log_var, z

class Decoder(layers.Layer):
	def __init__(self, original_dim, intermediate_dim=64, name='decoder', **kwargs):
		super(Decoder,self).__init__(name=name,**kwargs)
		self.dense_proj = layers.Dense(intermediate_dim,activation='relu')
		self.dense_output = layers.Dense(original_dim,activation='sigmoid')
	def call(self,inputs):
		x = self.dense_proj(inputs)
		return self.dense_output(x)

class VariationalAutoEncoder(tf.keras.Model):
	def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, name='autoencoder', **kwargs):
		super(VariationalAutoEncoder, self).__init__(name=name,**kwargs)
		self.original_dim = original_dim
		self.encoder = Encoder(latent_dim=latent_dim,intermediate_dim=intermediate_dim)
		self.decoder = Decoder(original_dim,intermediate_dim=intermediate_dim)
	def call(self, inputs):
		z_mean,z_log_var,z = self.encoder(inputs)
		reconstructed = self.decoder(z)
		kl_loss = -0.5*tf.reduce_mean(z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)
		self.add_loss(kl_loss)
		return reconstructed

original_dim = 784
vae = VariationalAutoEncoder(original_dim,64,32)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()
loss_metric = tf.keras.metrics.Mean()
(x_train,_),_ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float32')/255
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
# Iterate over epochs.
for epoch in range(3):
	for step,x_batch_train in enumerate(train_dataset):
		with tf.GradientTape() as tape:
			reconstructed = vae(x_batch_train)
			loss = mse_loss_fn(x_batch_train,reconstructed)
			loss += sum(vae.losses)
		grads = tape.gradient(loss,vae.trainable_variables)
		optimizer.apply_gradients(zip(grads,vae.trainable_variables))
		loss_metric(loss)
		if step%100 == 0: print(step,loss_metric.result())
# Built-in training loops.
vae = VariationalAutoEncoder(784,64,32)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer,loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train,x_train,epochs=3,batch_size=64)

# Beyond object-oriented development: the Functional API

original_dim = 784; intermediate_dim = 64; latent_dim = 32
original_inputs = tf.keras.Input(shape=(original_dim,),name='encoder_input')
x = layers.Dense(intermediate_dim,activation='relu')(original_inputs)
z_mean = layers.Dense(latent_dim,name='z_mean')(x)
z_log_var = layers.Dense(latent_dim,name='z_log_var')(x)
z = Sampling()((z_mean,z_log_var))
encoder = tf.keras.Model(inputs=original_inputs,outputs=z,name='encoder')
latent_inputs = tf.keras.Input(shape=(latent_dim,),name='z_sampling')
x = layers.Dense(intermediate_dim,activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim,activation='sigmoid')(x)
decoder = tf.keras.Model(inputs=latent_inputs,outputs=outputs,name='decoder')
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs,outputs=outputs,name='vae')
kl_loss = -0.5*tf.reduce_mean(z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)
vae.add_loss(kl_loss)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer,loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train,x_train,epochs=3,batch_size=64)
