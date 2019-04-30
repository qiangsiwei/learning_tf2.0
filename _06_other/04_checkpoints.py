# -*- coding: utf-8 -*-

import tensorflow as tf

class Net(tf.keras.Model):
	def __init__(self):
		super(Net,self).__init__()
		self.l1 = tf.keras.layers.Dense(5)
	def call(self, x):
		return self.l1(x)

def toy_dataset():
	inputs = tf.range(10.)[:,None]
	labels = inputs*5.+tf.range(5.)[None,:]
	return tf.data.Dataset.from_tensor_slices(
		dict(x=inputs,y=labels)).repeat(10).batch(2)

def train_step(net, example, optimizer):
	with tf.GradientTape() as tape:
		output = net(example['x'])
		loss = tf.reduce_mean(tf.abs(output-example['y']))
	variables = net.trainable_variables
	gradients = tape.gradient(loss,variables)
	optimizer.apply_gradients(zip(gradients,variables))
	return loss

net = Net()
opt = tf.keras.optimizers.Adam(0.1)
ckpt = tf.train.Checkpoint(step=tf.Variable(1),net=net,optimizer=opt)
manager = tf.train.CheckpointManager(ckpt,'./tf_ckpts',max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
for example in toy_dataset():
	loss = train_step(net,example,opt)
	ckpt.step.assign_add(1)
	if int(ckpt.step)%10 == 0:
		save_path = manager.save()

# Loading mechanics

to_restore = tf.Variable(tf.zeros([5]))
print(to_restore.numpy())
fake_layer = tf.train.Checkpoint(bias=to_restore)
fake_net = tf.train.Checkpoint(l1=fake_layer)
new_root = tf.train.Checkpoint(net=fake_net)
status = new_root.restore(tf.train.latest_checkpoint('./tf_ckpts/'))
print(to_restore.numpy())

# Delayed restorations

delayed_restore = tf.Variable(tf.zeros([1,5]))
print(delayed_restore.numpy())  # Not restored; still zeros
fake_layer.kernel = delayed_restore
print(delayed_restore.numpy())  # Restored

# Manually inspecting checkpoints

print(tf.train.list_variables(tf.train.latest_checkpoint('./tf_ckpts/')))

# List and dictionary tracking

save = tf.train.Checkpoint()
save.listed = [tf.Variable(1.)]
save.listed.append(tf.Variable(2.))
save.mapped = {'one':save.listed[0]}
save.mapped['two'] = save.listed[1]
save_path = save.save('./tf_list_example')
restore = tf.train.Checkpoint()
v2 = tf.Variable(0.)
assert 0. == v2.numpy()
restore.mapped = {'two':v2}
restore.restore(save_path)
assert 2. == v2.numpy()
restore.listed = []
print(restore.listed)
v1 = tf.Variable(0.)
restore.listed.append(v1)
assert 1. == v1.numpy()

# Saving object-based checkpoints with Estimator

import tensorflow.compat.v1 as tf_compat

def model_fn(features, labels, mode):
	net = Net()
	opt = tf.keras.optimizers.Adam(0.1)
	ckpt = tf.train.Checkpoint(step=tf_compat.train.get_global_step(),net=net,optimizer=opt)
	with tf.GradientTape() as tape:
		output = net(features['x'])
		loss = tf.reduce_mean(tf.abs(output-features['y']))
	variables = net.trainable_variables
	gradients = tape.gradient(loss,variables)
	return tf.estimator.EstimatorSpec(mode,loss=loss,
		train_op=tf.group(opt.apply_gradients(zip(gradients,variables)),ckpt.step.assign_add(1)),
		scaffold=tf_compat.train.Scaffold(saver=ckpt))
tf.keras.backend.clear_session()
est = tf.estimator.Estimator(model_fn,'./tf_estimator_example/')
est.train(toy_dataset,steps=10)
net = Net()
opt = tf.keras.optimizers.Adam(0.1)
ckpt = tf.train.Checkpoint(step=tf.Variable(1,dtype=tf.int64),net=net,optimizer=opt)
ckpt.restore(tf.train.latest_checkpoint('./tf_estimator_example/'))
ckpt.step.numpy()
