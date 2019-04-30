# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

file = tf.keras.utils.get_file('grace_hopper.jpg',
	'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
img = tf.keras.preprocessing.image.load_img(file,target_size=[224,224])
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(np.array(img)[tf.newaxis,...])
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
	'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
pretrained_model = tf.keras.applications.MobileNet()
result_before_save = pretrained_model(x)
decoded = imagenet_labels[np.argsort(result_before_save)[0,::-1][:5]+1]
print(decoded)
tf.saved_model.save(pretrained_model,'/tmp/mobilenet/1/')
loaded = tf.saved_model.load('/tmp/mobilenet/1/')
print(list(loaded.signatures.keys()))
infer = loaded.signatures['serving_default']
print(infer.structured_outputs)
labeling = infer(tf.constant(x))['reshape_2']
decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]
print(decoded)

# ----- Serving the model ----- 

# nohup tensorflow_model_server \
#   --rest_api_port=8501 \
#   --model_name=mobilenet \
#   --model_base_path='/tmp/mobilenet' >server.log 2>&1

# import json
# import numpy
# import requests
# data = json.dumps({'signature_name':'serving_default',
#                    'instances':x.tolist()})
# headers = {'content-type':'application/json'}
# json_response = requests.post('http://localhost:8501/v1/models/mobilenet:predict',
#                               data=data,headers=headers)
# predictions = numpy.array(json.loads(json_response.text)['predictions'])

# ----- SavedModel format -----

# assets  saved_model.pb  variables
# variables.data-00000-of-00001  variables.index

# ----- Exporting custom models -----

class CustomModule(tf.Module):
	def __init__(self):
		super(CustomModule,self).__init__()
		self.v = tf.Variable(1.)
	@tf.function
	def __call__(self, x):
		return x*self.v
	@tf.function(input_signature=[tf.TensorSpec([],tf.float32)])
	def mutate(self, new_v):
		self.v.assign(new_v)
module = CustomModule()
module(tf.constant(0.))
tf.saved_model.save(module,'/tmp/module_no_signatures')
imported = tf.saved_model.load('/tmp/module_no_signatures')
assert 3. == imported(tf.constant(3.)).numpy()
imported.mutate(tf.constant(2.))
assert 6. == imported(tf.constant(3.)).numpy()
module.__call__.get_concrete_function(x=tf.TensorSpec([None],tf.float32))
tf.saved_model.save(module,'/tmp/module_no_signatures')
imported = tf.saved_model.load('/tmp/module_no_signatures')
assert [3.] == imported(tf.constant([3.])).numpy()

# Identifying a signature to export

call = module.__call__.get_concrete_function(tf.TensorSpec(None,tf.float32))
tf.saved_model.save(module,'/tmp/module_with_signature',signatures=call)

imported = tf.saved_model.load('/tmp/module_with_signature')
signature = imported.signatures['serving_default']
assert [3.] == signature(x=tf.constant([3.]))['output_0'].numpy()
imported.mutate(tf.constant(2.))
assert [6.] == signature(x=tf.constant([3.]))['output_0'].numpy()
assert 2. == imported.v.numpy()

@tf.function(input_signature=[tf.TensorSpec([],tf.string)])
def parse_string(string_input):
	return imported(tf.strings.to_number(string_input))
signatures = {'serving_default':parse_string,
              'from_float':imported.signatures['serving_default']}
tf.saved_model.save(imported,'/tmp/module_with_multiple_signatures',signatures)

# ----- Fine-tuning imported models ----- 

optimizer = tf.optimizers.SGD(0.05)
def train_step():
	with tf.GradientTape() as tape:
		loss = (10.-imported(tf.constant(2.)))**2
	variables = tape.watched_variables()
	grads = tape.gradient(loss,variables)
	optimizer.apply_gradients(zip(grads,variables))
	return loss
for _ in range(10):
	print(train_step(),imported.v.numpy())

# Control flow in SavedModels

# ----- SavedModels from Estimators ----- 

input_column = tf.feature_column.numeric_column('x')
estimator = tf.estimator.LinearClassifier(feature_columns=[input_column])
def input_fn():
	return tf.data.Dataset.from_tensor_slices(({'x':[1.,2.,3.,4.]},[1,1,0,0])).repeat(200).shuffle(64).batch(16)
estimator.train(input_fn)
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
	tf.feature_column.make_parse_example_spec([input_column]))
export_path = estimator.export_saved_model('/tmp/from_estimator/',serving_input_fn)
imported = tf.saved_model.load(export_path)
def predict(x):
	example = tf.train.Example()
	example.features.feature['x'].float_list.value.extend([x])
	return imported.signatures['predict'](examples=tf.constant([example.SerializeToString()]))
print(predict(1.5))

# ----- Load a SavedModel in C++ ----- 

# ----- Details of the SavedModel command line interface ----- 
