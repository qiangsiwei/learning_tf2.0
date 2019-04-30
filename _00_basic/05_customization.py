# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import os, timeit, tempfile
import numpy as np
import tensorflow as tf

# ----- tensors and operations ------

print(tf.add(1,2))
print(tf.add([1,2],[3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))
print(tf.square(2)+tf.square(3))
x = tf.matmul([[1]],[[2,3]])
print(x,x.shape,x.dtype)
ndarray = np.ones([3, 3])
tensor = tf.multiply(ndarray,42)
print(tensor)
print(np.add(tensor,1))
print(tensor.numpy())
x = tf.random.uniform([3, 3])
print(tf.test.is_gpu_available())
print(x.device.endswith('GPU:0'))

ds_tensors = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])
_,filename = tempfile.mkstemp()
with open(filename,'w') as f:
  f.write('Line 1\nLine 2\nLine 3\nLine 4')
ds_file = tf.data.TextLineDataset(filename)
print(ds_file)
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)
print ds_file
for x in ds_tensors: print(x)
for x in ds_file: print(x)

# ----- custom layers ------

layer = tf.keras.layers.Dense(100)
layer = tf.keras.layers.Dense(10,input_shape=(None,5))
print(layer(tf.zeros([10,5])))
print(layer.variables,layer.kernel,layer.bias)

class MyDenseLayer(tf.keras.layers.Layer):
	def __init__(self, num_outputs):
		super(MyDenseLayer, self).__init__()
		self.num_outputs = num_outputs
	def build(self, input_shape):
		self.kernel = self.add_variable('kernel',shape=[int(input_shape[-1]),self.num_outputs])
	def call(self, input):
		return tf.matmul(input, self.kernel)
layer = MyDenseLayer(10)
print(layer(tf.zeros([10,5])))
print(layer.trainable_variables)

class ResnetIdentityBlock(tf.keras.Model):
	def __init__(self, kernel_size, filters):
		super(ResnetIdentityBlock, self).__init__(name='')
		filters1,filters2,filters3 = filters
		self.conv2a = tf.keras.layers.Conv2D(filters1,(1,1))
		self.bn2a = tf.keras.layers.BatchNormalization()
		self.conv2b = tf.keras.layers.Conv2D(filters2,kernel_size,padding='same')
		self.bn2b = tf.keras.layers.BatchNormalization()
		self.conv2c = tf.keras.layers.Conv2D(filters3,(1,1))
		self.bn2c = tf.keras.layers.BatchNormalization()
	def call(self, input_tensor, training=False):
		x = self.conv2a(input_tensor)
		x = self.bn2a(x,training=training)
		x = tf.nn.relu(x)
		x = self.conv2b(x)
		x = self.bn2b(x,training=training)
		x = tf.nn.relu(x)
		x = self.conv2c(x)
		x = self.bn2c(x,training=training)
		x += input_tensor
		return tf.nn.relu(x)
block = ResnetIdentityBlock(1,[1,2,3])
print(block(tf.zeros([1,2,3,3])))
print([x.name for x in block.trainable_variables])

my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1,(1,1),input_shape=(None,None,3)),
							  tf.keras.layers.BatchNormalization(),
							  tf.keras.layers.Conv2D(2,1,padding='same'),
							  tf.keras.layers.BatchNormalization(),
							  tf.keras.layers.Conv2D(3,(1,1)),
							  tf.keras.layers.BatchNormalization()])
print(my_seq(tf.zeros([1,2,3,3])))

# ----- automatic differentiation ------

x = tf.ones((2, 2))
with tf.GradientTape(persistent=True) as t:
	t.watch(x)
	y = tf.reduce_sum(x)
	z = tf.multiply(y,y)
dz_dx = t.gradient(z,x)
for i in [0,1]:
	for j in [0,1]:
		assert dz_dx[i][j].numpy() == 8.0
dz_dy = t.gradient(z,y)
assert dz_dy.numpy() == 8.0

def f(x, y):
	output = 1.0
	for i in range(y):
		if i > 1 and i < 5:
			output = tf.multiply(output,x)
	return output
def grad(x, y):
	with tf.GradientTape() as t:
		t.watch(x)
		out = f(x,y)
	return t.gradient(out,x)
x = tf.convert_to_tensor(2.0)
assert grad(x,6).numpy() == 12.0
assert grad(x,5).numpy() == 12.0
assert grad(x,4).numpy() == 4.0

x = tf.Variable(1.0)
with tf.GradientTape() as t:
	with tf.GradientTape() as t2:
		y = x*x*x
	dy_dx = t2.gradient(y,x)
d2y_dx2 = t.gradient(dy_dx,x)
assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0

# ----- custom training: basics ------

class Model(object):
	def __init__(self):
		self.W = tf.Variable(5.0)
		self.b = tf.Variable(0.0)
	def __call__(self, x):
		return self.W*x+self.b
model = Model()
assert model(3.0).numpy() == 15.0
def loss(predicted_y, desired_y):
	return tf.reduce_mean(tf.square(predicted_y-desired_y))
TRUE_W = 3.0; TRUE_b = 2.0; NUM_EXAMPLES = 1000
inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs*TRUE_W+TRUE_b+noise
def train(model, inputs, outputs, learning_rate):
	with tf.GradientTape() as t:
		current_loss = loss(model(inputs),outputs)
	dW,db = t.gradient(current_loss,[model.W,model.b])
	model.W.assign_sub(learning_rate*dW)
	model.b.assign_sub(learning_rate*db)
model = Model()
for epoch in range(10):
	current_loss = loss(model(inputs),outputs)
	train(model,inputs,outputs,learning_rate=0.1)
	print(epoch,model.W.numpy(),model.b.numpy(),current_loss)

train_url = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv'
train_fp = tf.keras.utils.get_file(fname=os.path.basename(train_url),origin=train_url)
column_names = ['sepal_length','sepal_width','petal_length','petal_width','species']
train_dataset = tf.data.experimental.make_csv_dataset(train_fp,32,\
	column_names=column_names,label_name=column_names[-1],num_epochs=1)
def pack_features_vector(features, labels):
	features = tf.stack(list(features.values()),axis=1)
	return features,labels
train_dataset = train_dataset.map(pack_features_vector)
model = tf.keras.Sequential([
	tf.keras.layers.Dense(10,activation=tf.nn.relu,input_shape=(4,)),
	tf.keras.layers.Dense(10,activation=tf.nn.relu),
	tf.keras.layers.Dense(3)])
def loss(model, x, y):
	return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true=y,y_pred=model(x))
def grad(model, inputs, targets):
	with tf.GradientTape() as tape:
		loss_value = loss(model,inputs,targets)
	return loss_value,tape.gradient(loss_value,model.trainable_variables)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
for epoch in range(200):
	epoch_loss_avg = tf.keras.metrics.Mean()
	epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
	for x,y in train_dataset:
		loss_value,grads = grad(model,x,y)
		optimizer.apply_gradients(zip(grads,model.trainable_variables))
		epoch_loss_avg(loss_value)
		epoch_accuracy(y,model(x))
	if (epoch+1)%50 == 0:
		print(epoch,epoch_loss_avg.result(),epoch_accuracy.result())
test_url = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv'
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),origin=test_url)
test_dataset = tf.data.experimental.make_csv_dataset(test_fp,32,
	column_names=column_names,label_name=column_names[-1],num_epochs=1,shuffle=False)
test_dataset = test_dataset.map(pack_features_vector)
test_accuracy = tf.keras.metrics.Accuracy()
for x,y in test_dataset:
	logits = model(x)
	prediction = tf.argmax(logits,axis=1,output_type=tf.int32)
	test_accuracy(prediction,y)
print(test_accuracy.result())
# print(tf.stack([y,prediction],axis=1))

# ----- tf function and autograph ------

@tf.function
def add(a, b):
  return a+b
print(add(tf.ones([2,2]),tf.ones([2,2])))
v = tf.Variable(1.0)
with tf.GradientTape() as tape:
	result = add(v,1.0)
tape.gradient(result,v)

@tf.function
def dense_layer(x, w, b):
	return add(tf.matmul(x,w),b)
dense_layer(tf.ones([3,2]),tf.ones([2,2]),tf.ones([2]))

# polymorphism

@tf.function
def add(a):
	return a+a
print(add(1))
print(add(1.1))
print(add(tf.constant('a')))
c = add.get_concrete_function(tf.TensorSpec(shape=None,dtype=tf.string))
c(a=tf.constant('a'))

# functions can be faster than eager code, for graphs with many small ops

conv_layer = tf.keras.layers.Conv2D(100,3)
@tf.function
def conv_fn(image):
	return conv_layer(image)
image = tf.zeros([1,200,200,100])
conv_layer(image); conv_fn(image)
print(timeit.timeit(lambda:conv_layer(image),number=10))
print(timeit.timeit(lambda:conv_fn(image),number=10))
lstm_cell = tf.keras.layers.LSTMCell(10)
@tf.function
def lstm_fn(input, state):
	return lstm_cell(input,state)
input = tf.zeros([10,10])
state = [tf.zeros([10,10])]*2
lstm_cell(input, state); lstm_fn(input, state)
print(timeit.timeit(lambda:lstm_cell(input,state),number=10))
print(timeit.timeit(lambda:lstm_fn(input,state),number=10))

a = tf.Variable(1.0)
b = tf.Variable(2.0)
@tf.function
def f(x, y):
	a.assign(y*b)
	b.assign_add(x*a)
	return a+b
f(1.0,2.0)

v = tf.Variable(1.0)
@tf.function
def f(x):
	v.assign_add(x)
	return v
print(f(1.),f(2.0))

class C: pass
obj = C(); obj.v = None
@tf.function
def g(x):
	if obj.v is None:
		obj.v = tf.Variable(1.0)
	return obj.v.assign_add(x)
print(g(1.),g(2.0))

state = []
@tf.function
def fn(x):
	if not state:
		state.append(tf.Variable(2.0*x))
		state.append(tf.Variable(state[0]*3.0))
	return state[0]*x*state[1]
print(fn(tf.constant(1.0)),fn(tf.constant(3.0)))

@tf.function
def f(x):
	while tf.reduce_sum(x) > 1:
		tf.print(x)
		x = tf.tanh(x)
	return x
f(tf.random.uniform([10]))

def f(x):
	while tf.reduce_sum(x) > 1:
		tf.print(x)
		x = tf.tanh(x)
	return x
print(tf.autograph.to_code(f))

@tf.function
def f(x):
	for i in tf.range(10):
		tf.print(i)
		tf.Assert(i<10,['a'])
		x += x
	return x

@tf.function
def f(x):
	ta = tf.TensorArray(tf.float32, size=10)
	for i in tf.range(10):
		x += x
		ta = ta.write(i, x)
	return ta.stack()
f(10.0)

