# -*- coding: utf-8 -*-

import os, json
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

annotation_zip = tf.keras.utils.get_file('captions.zip',cache_subdir=os.path.abspath('.'),
	origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',extract=True)
annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
name_of_zip = 'train2014.zip'
if not os.path.exists(os.path.abspath('.')+'/'+name_of_zip):
	image_zip = tf.keras.utils.get_file(name_of_zip,cache_subdir=os.path.abspath('.'),
		origin='http://images.cocodataset.org/zips/train2014.zip',extract=True)
PATH = os.path.abspath('.')+'/train2014/'

all_img_name_vector,all_captions = [],[]
with open(annotation_file,'r') as f:
	for annot in json.load(f)['annotations']:
		caption = '<start> '+annot['caption']+' <end>'
		image_id = annot['image_id']
		full_coco_image_path = PATH+'COCO_train2014_'+'%012d.jpg'%(image_id)
		all_img_name_vector.append(full_coco_image_path)
		all_captions.append(caption)
img_name_vector,train_captions = shuffle(all_img_name_vector,all_captions,random_state=1)
num_examples = 30000
img_name_vector,train_captions = img_name_vector[:num_examples],train_captions[:num_examples]

def load_image(image_path):
	img = tf.io.read_file(image_path)
	img = tf.image.decode_jpeg(img,channels=3)
	img = tf.image.resize(img,(299,299))
	img = tf.keras.applications.inception_v3.preprocess_input(img)
	return img, image_path

image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
image_features_extract_model = tf.keras.Model(image_model.input,image_model.layers[-1].output)

encode_train = sorted(set(img_name_vector))
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(load_image,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
for img,path in image_dataset:
	batch_features = image_features_extract_model(img)
	batch_features = tf.reshape(batch_features,(batch_features.shape[0],-1,batch_features.shape[3]))
	for bf,p in zip(batch_features,path):
		path_of_feature = p.numpy().decode('utf-8')
		np.save(path_of_feature,bf.numpy())

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000,oov_token='<unk>',filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs,padding='post')
max_length = max(len(t) for t in train_seqs)

img_name_train,img_name_val,cap_train,cap_val = train_test_split(img_name_vector,cap_vector,test_size=0.2,random_state=0)
BATCH_SIZE = 64; BUFFER_SIZE = 1000; embedding_dim = 256; units = 512
vocab_size = len(tokenizer.word_index)+1
num_steps = len(img_name_train)//BATCH_SIZE
features_shape = 2048; attention_features_shape = 64
def map_func(img_name, cap):
	return np.load(img_name.decode('utf-8')+'.npy'),cap
dataset = tf.data.Dataset.from_tensor_slices((img_name_train,cap_train))
dataset = dataset.map(lambda item1,item2:tf.numpy_function(
          map_func,[item1,item2],[tf.float32,tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

class CNN_Encoder(tf.keras.Model):
	def __init__(self, embedding_dim):
		super(CNN_Encoder,self).__init__()
		self.fc = tf.keras.layers.Dense(embedding_dim)
	def call(self, x):
		x = self.fc(x)
		x = tf.nn.relu(x)
		return x
encoder = CNN_Encoder(embedding_dim)

class BahdanauAttention(tf.keras.Model):
	def __init__(self, units):
		super(BahdanauAttention,self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)
	def call(self, features, hidden):
		hidden_with_time_axis = tf.expand_dims(hidden,1)
		score = tf.nn.tanh(self.W1(features)+self.W2(hidden_with_time_axis))
		attention_weights = tf.nn.softmax(self.V(score),axis=1)
		context_vector = attention_weights*features
		context_vector = tf.reduce_sum(context_vector,axis=1)
		return context_vector, attention_weights

class RNN_Decoder(tf.keras.Model):
	def __init__(self, embedding_dim, units, vocab_size):
		super(RNN_Decoder,self).__init__()
		self.units = units
		self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim)
		self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.fc2 = tf.keras.layers.Dense(vocab_size)
		self.attention = BahdanauAttention(self.units)
	def call(self, x, features, hidden):
		context_vector,attention_weights = self.attention(features,hidden)
		x = self.embedding(x)
		x = tf.concat([tf.expand_dims(context_vector,1),x],axis=-1)
		output,state = self.gru(x)
		x = self.fc1(output)
		x = tf.reshape(x,(-1,x.shape[2]))
		x = self.fc2(x)
		return x, state, attention_weights
	def reset_state(self, batch_size):
		return tf.zeros((batch_size, self.units))
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.keras.optimizers.Adam()
def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real,0))
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')(real,pred)
	return tf.reduce_mean(loss*tf.cast(mask,dtype=loss.dtype))

checkpoint_path = './checkpoints/train'
ckpt = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)
ckpt_manager = tf.train.CheckpointManager(ckpt,checkpoint_path,max_to_keep=5)

start_epoch = 0 if not ckpt_manager.latest_checkpoint else\
	int(ckpt_manager.latest_checkpoint.split('-')[-1])

@tf.function
def train_step(img_tensor, target):
	loss = 0
	with tf.GradientTape() as tape:
		hidden = decoder.reset_state(batch_size=target.shape[0])
		dec_input = tf.expand_dims([tokenizer.word_index['<start>']]*BATCH_SIZE,1)
		for t in range(1,target.shape[1]):
			predictions,hidden,_ = decoder(dec_input,encoder(img_tensor),hidden)
			loss += loss_function(target[:,t],predictions)
			dec_input = tf.expand_dims(target[:,t],1)
	batch_loss = (loss/int(target.shape[1]))
	variables = encoder.trainable_variables+decoder.trainable_variables
	gradients = tape.gradient(loss,variables)
	optimizer.apply_gradients(zip(gradients,variables))
	return batch_loss

for epoch in range(start_epoch,20):
	total_loss = 0
	for batch,(img_tensor,target) in enumerate(dataset):
		batch_loss = train_step(img_tensor,target)
		if (batch+1)%100 == 0:
			print(epoch+1,batch,batch_loss.numpy())
	if (epoch+1)%5 == 0:
		ckpt_manager.save()

def evaluate(image):
    # attention_plot = np.zeros((max_length,attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0],0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val,(img_tensor_val.shape[0],-1,img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']],0)
    result = []
    for t in range(max_length):
        predictions,hidden,attention_weights = decoder(dec_input,features,hidden)
        # attention_plot[t] = tf.reshape(attention_weights,(-1,)).numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            return result # attention_plot
        dec_input = tf.expand_dims([predicted_id],0)
    # attention_plot = attention_plot[:len(result),:]
    return result # attention_plot

def plot_attention(image, result, attention_plot):
	temp_image = np.array(Image.open(image))
	fig = plt.figure(figsize=(10,10))
	len_result = len(result)
	for l in range(len_result):
		temp_att = np.resize(attention_plot[l],(8,8))
		ax = fig.add_subplot(len_result//2,len_result//2,l+1)
		ax.set_title(result[l])
		img = ax.imshow(temp_image)
		ax.imshow(temp_att,cmap='gray',alpha=0.6,extent=img.get_extent())
	plt.tight_layout()
	plt.show()

rid = np.random.randint(0,len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result = evaluate(image) # attention_plot
print(real_caption,'\n',' '.join(result))
# plot_attention(image_path, result, attention_plot)
# Image.open(image_path)
