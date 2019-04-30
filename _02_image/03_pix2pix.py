# -*- coding: utf-8 -*-

import os
import tensorflow as tf

path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
	origin='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz',extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip),'facades/')

BUFFER_SIZE = 400; BATCH_SIZE = 1; IMG_WIDTH = 256; IMG_HEIGHT = 256

def load(image_file):
	image = tf.io.read_file(image_file)
	image = tf.image.decode_jpeg(image)
	w = tf.shape(image)[1]; w = w//2
	real_image = image[:,:w,:]
	input_image = image[:,w:,:]
	real_image = tf.cast(real_image,tf.float32)
	input_image = tf.cast(input_image,tf.float32)
	return input_image, real_image

def resize(input_image, real_image, height, width):
	real_image = tf.image.resize(real_image,[height,width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	input_image = tf.image.resize(input_image,[height,width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	return input_image, real_image

def random_crop(input_image, real_image):
	stacked_image = tf.stack([input_image,real_image],axis=0)
	cropped_image = tf.image.random_crop(stacked_image,size=[2,IMG_HEIGHT,IMG_WIDTH,3])
	return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
	real_image = (real_image/127.5)-1
	input_image = (input_image/127.5)-1
	return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
	input_image,real_image = resize(input_image,real_image,286,286)
	input_image,real_image = random_crop(input_image,real_image)
	if tf.random.uniform(()) > 0.5:
		real_image = tf.image.flip_left_right(real_image)
		input_image = tf.image.flip_left_right(input_image)
	return input_image, real_image

def load_image_train(image_file):
	input_image,real_image = load(image_file)
	input_image,real_image = random_jitter(input_image,real_image)
	input_image,real_image = normalize(input_image,real_image)
	return input_image, real_image

def load_image_test(image_file):
	input_image,real_image = load(image_file)
	input_image,real_image = resize(input_image,real_image,IMG_HEIGHT,IMG_WIDTH)
	input_image,real_image = normalize(input_image,real_image)
	return input_image, real_image

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.map(load_image_train,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(1)

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(1)

def downsample(filters, size, apply_batchnorm=True):
	initializer = tf.random_normal_initializer(0.,0.02)
	result = tf.keras.Sequential()
	result.add(tf.keras.layers.Conv2D(filters,size,strides=2,
		padding='same',kernel_initializer=initializer,use_bias=False))
	if apply_batchnorm:
		result.add(tf.keras.layers.BatchNormalization())
	result.add(tf.keras.layers.LeakyReLU())
	return result

def upsample(filters, size, apply_dropout=False):
	initializer = tf.random_normal_initializer(0., 0.02)
	result = tf.keras.Sequential()
	result.add(tf.keras.layers.Conv2DTranspose(filters,size,strides=2,
		padding='same',kernel_initializer=initializer,use_bias=False))
	result.add(tf.keras.layers.BatchNormalization())
	if apply_dropout:
		result.add(tf.keras.layers.Dropout(0.5))
	result.add(tf.keras.layers.ReLU())
	return result

def Generator(OUTPUT_CHANNELS=3):
	down_stack = [
		downsample(64,4,apply_batchnorm=False),
		downsample(128,4), # (bs, 64, 64, 128)
		downsample(256,4), # (bs, 32, 32, 256)
		downsample(512,4), # (bs, 16, 16, 512)
		downsample(512,4), # (bs, 8, 8, 512)
		downsample(512,4), # (bs, 4, 4, 512)
		downsample(512,4), # (bs, 2, 2, 512)
		downsample(512,4)] # (bs, 1, 1, 512)
	up_stack = [
		upsample(512,4,apply_dropout=True), # (bs, 2, 2, 1024)
		upsample(512,4,apply_dropout=True), # (bs, 4, 4, 1024)
		upsample(512,4,apply_dropout=True), # (bs, 8, 8, 1024)
		upsample(512,4), # (bs, 16, 16, 1024)
		upsample(256,4), # (bs, 32, 32, 512)
		upsample(128,4), # (bs, 64, 64, 256)
		upsample(64,4)] # (bs, 128, 128, 128)
	initializer = tf.random_normal_initializer(0.,0.02)
	last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS,4,strides=2,
		padding='same',kernel_initializer=initializer,activation='tanh') # (bs, 256, 256, 3)
	concat = tf.keras.layers.Concatenate()
	inputs = tf.keras.layers.Input(shape=[None,None,3])
	x = inputs
	skips = []
	for down in down_stack:
		x = down(x)
		skips.append(x)
	skips = reversed(skips[:-1])
	for up,skip in zip(up_stack,skips):
		x = up(x)
		x = concat([x,skip])
	x = last(x)
	return tf.keras.Model(inputs=inputs,outputs=x)
generator = Generator()

def Discriminator():
	initializer = tf.random_normal_initializer(0.,0.02)
	inp = tf.keras.layers.Input(shape=[None,None,3],name='input_image')
	tar = tf.keras.layers.Input(shape=[None,None,3],name='target_image')
	x = tf.keras.layers.concatenate([inp,tar])
	down1 = downsample(64,4,False)(x)
	down2 = downsample(128,4)(down1)
	down3 = downsample(256,4)(down2)
	zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
	conv = tf.keras.layers.Conv2D(512,4,strides=1,
		kernel_initializer=initializer,use_bias=False)(zero_pad1)
	batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
	leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
	zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
	last = tf.keras.layers.Conv2D(1,4,strides=1,
		kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
	return tf.keras.Model(inputs=[inp,tar],outputs=last)
discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
	real_loss = loss_object(tf.ones_like(disc_real_output),disc_real_output)
	generated_loss = loss_object(tf.zeros_like(disc_generated_output),disc_generated_output)
	total_disc_loss = real_loss+generated_loss
	return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
	gan_loss = loss_object(tf.ones_like(disc_generated_output),disc_generated_output)
	l1_loss = tf.reduce_mean(tf.abs(target-gen_output))
	total_gen_loss = gan_loss+(LAMBDA*l1_loss)
	return total_gen_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(generator=generator,
								 discriminator=discriminator,
								 generator_optimizer=generator_optimizer,
								 discriminator_optimizer=discriminator_optimizer)

@tf.function
def train_step(input_image, target):
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		gen_output = generator(input_image,training=True)
		disc_real_output = discriminator([input_image,target],training=True)
		disc_generated_output = discriminator([input_image,gen_output],training=True)
		gen_loss = generator_loss(disc_generated_output,gen_output,target)
		disc_loss = discriminator_loss(disc_real_output,disc_generated_output)
	generator_gradients = gen_tape.gradient(gen_loss,generator.trainable_variables)
	discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
	generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))

def generate_images(model, epoch, test_input, tar):
	prediction = model(test_input,training=True)
	plt.figure(figsize=(15,15))
	display_list = [test_input[0],tar[0],prediction[0]]
	title = ['Input Image','Ground Truth','Predicted Image']
	for i in range(3):
		plt.subplot(1,3,i+1)
		plt.title(title[i])
		plt.imshow(display_list[i]*0.5+0.5)
		plt.axis('off')
	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

def train(dataset, epochs):
	for epoch in range(epochs):
		for input_image,target in dataset:
			train_step(input_image,target)
		for inp,tar in test_dataset.take(1):
			generate_images(generator,epoch,inp,tar)
		if (epoch+1)%20 == 0:
			checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 200
train(train_dataset,EPOCHS)
