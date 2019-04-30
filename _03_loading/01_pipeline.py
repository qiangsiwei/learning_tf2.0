# -*- coding: utf-8 -*-

import time, pathlib
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_root_orig = tf.keras.utils.get_file(
	origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
	fname='flower_photos',untar=True)
data_root = pathlib.Path(data_root_orig)

all_image_paths = [str(path) for path in list(data_root.glob('*/*'))]
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name,index) for index,name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
image_count = len(all_image_paths)

img_raw = tf.io.read_file(all_image_paths[0])
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape,img_tensor.dtype)
img_final = tf.image.resize(img_tensor,[192,192])/255.0

# Build a tf.data.Dataset
def preprocess_image(image):
	image = tf.image.decode_jpeg(image,channels=3)
	image = tf.image.resize(image,[192,192])
	image /= 255.0
	return image
def load_and_preprocess_image(path):
	image = tf.io.read_file(path)
	return preprocess_image(image)
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels,tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

# Alternative
ds = tf.data.Dataset.from_tensor_slices((all_image_paths,all_image_labels))
def load_and_preprocess_from_path_label(path, label):
	return load_and_preprocess_image(path),label
image_label_ds = ds.map(load_and_preprocess_from_path_label)

# Basic methods for training
BATCH_SIZE = 32
ds = image_label_ds.shuffle(buffer_size=image_count).repeat().batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3),include_top=False)
mobile_net.trainable = False
keras_ds = ds.map(lambda image,label:(2*image-1,label))
model = tf.keras.Sequential([
	mobile_net,
	tf.keras.layers.GlobalAveragePooling2D(),
	tf.keras.layers.Dense(len(label_names))])
model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(ds,epochs=1,steps_per_epoch=3)

# Performance
steps_per_epoch=len(all_image_paths)/BATCH_SIZE
def timeit(ds, steps=2*steps_per_epoch+1):
	overall_start = time.time()
	it = iter(ds.take(steps+1)); next(it)
	start = time.time()
	for i,(images,labels) in enumerate(it):
		if i%10 == 0: print '.',
	end = time.time()
	duration = end-start
	print('Total time: {}s'.format(end-overall_start))
# ds = image_label_ds.cache()
# ds = image_label_ds.cache(filename='./cache.tf-data')
ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
timeit(ds)

image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(image_ds)
image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)

paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = paths_ds.map(load_and_preprocess_image)
ds = image_ds.map(tf.io.serialize_tensor)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(ds)
ds = tf.data.TFRecordDataset('images.tfrec')
def parse(x):
	result = tf.io.parse_tensor(x,out_type=tf.float32)
	return tf.reshape(result,[192,192,3])
ds = ds.map(parse,num_parallel_calls=AUTOTUNE)
