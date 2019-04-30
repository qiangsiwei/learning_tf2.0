# -*- coding: utf-8 -*-

import io, os, re, time, unicodedata
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

path_to_zip = tf.keras.utils.get_file('spa-eng.zip',origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',extract=True)
path_to_file = os.path.dirname(path_to_zip)+'/spa-eng/spa.txt'

def unicode_to_ascii(s):
	return ''.join(c for c in unicodedata.normalize('NFD',s)
		if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
	w = unicode_to_ascii(w.lower().strip())
	w = re.sub(r'[^a-zA-Z?.!,¿]+',' ',re.sub(r'[" "]+',' ',re.sub(r'([?.!,¿])',r' \1 ',w)))
	return '<start> '+w.rstrip().strip()+' <end>'

def create_dataset(path, num_examples):
	lines = io.open(path,encoding='UTF-8').read().strip().split('\n')
	word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
	return zip(*word_pairs)

en,sp = create_dataset(path_to_file,None)

def max_length(tensor):
	return max(len(t) for t in tensor)

def tokenize(lang):
	lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
	lang_tokenizer.fit_on_texts(lang)
	tensor = lang_tokenizer.texts_to_sequences(lang)
	tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')
	return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
	tar_lang,inp_lang = create_dataset(path,num_examples)
	inp_tensor,inp_lang_tokenizer = tokenize(inp_lang)
	tar_tensor,tar_lang_tokenizer = tokenize(tar_lang)
	return inp_tensor, tar_tensor, inp_lang_tokenizer, tar_lang_tokenizer

num_examples = 30000
inp_tensor,tar_tensor,inp_lang,tar_lang = load_dataset(path_to_file,num_examples)
max_length_tar,max_length_inp = max_length(tar_tensor),max_length(inp_tensor)
inp_tensor_train,inp_tensor_val,tar_tensor_train,tar_tensor_val = train_test_split(inp_tensor,tar_tensor,test_size=0.2)

BUFFER_SIZE = len(inp_tensor_train); BATCH_SIZE = 64
steps_per_epoch = len(inp_tensor_train)//BATCH_SIZE
embedding_dim = 256; units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(tar_lang.word_index)+1
dataset = tf.data.Dataset.from_tensor_slices((inp_tensor_train,tar_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)

class Encoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
		super(Encoder, self).__init__()
		self.batch_sz = batch_sz
		self.enc_units = enc_units
		self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim)
		self.gru = tf.keras.layers.GRU(self.enc_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
	def call(self, x, hidden):
		x = self.embedding(x)
		output,state = self.gru(x,initial_state=hidden)
		return output,state
	def initialize_hidden_state(self):
		return tf.zeros((self.batch_sz,self.enc_units))
encoder = Encoder(vocab_inp_size,embedding_dim,units,BATCH_SIZE)

class BahdanauAttention(tf.keras.Model):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)
	def call(self, query, values):
		hidden_with_time_axis = tf.expand_dims(query,1)
		score = self.V(tf.nn.tanh(self.W1(values)+self.W2(hidden_with_time_axis)))
		attention_weights = tf.nn.softmax(score,axis=1)
		context_vector = attention_weights*values
		context_vector = tf.reduce_sum(context_vector,axis=1)
		return context_vector, attention_weights

class Decoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
		super(Decoder, self).__init__()
		self.batch_sz = batch_sz
		self.dec_units = dec_units
		self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim)
		self.gru = tf.keras.layers.GRU(self.dec_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
		self.fc = tf.keras.layers.Dense(vocab_size)
		self.attention = BahdanauAttention(self.dec_units)
	def call(self, x, hidden, enc_output):
		context_vector,attention_weights = self.attention(hidden,enc_output)
		x = self.embedding(x)
		x = tf.concat([tf.expand_dims(context_vector,1),x],axis=-1)
		output,state = self.gru(x)
		output = tf.reshape(output,(-1,output.shape[2]))
		x = self.fc(output)
		return x, state, attention_weights
decoder = Decoder(vocab_tar_size,embedding_dim,units,BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real,0))
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')(real,pred)
	return tf.reduce_mean(loss*tf.cast(mask,dtype=loss.dtype))

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

@tf.function
def train_step(inp, tar, enc_hidden):
	loss = 0
	with tf.GradientTape() as tape:
		enc_output,enc_hidden = encoder(inp,enc_hidden)
		dec_hidden = enc_hidden
		dec_input = tf.expand_dims([tar_lang.word_index['<start>']]*BATCH_SIZE,1)
		for t in range(1,tar.shape[1]):
			predictions,dec_hidden,_ = decoder(dec_input,dec_hidden,enc_output)
			loss += loss_function(tar[:,t],predictions)
			dec_input = tf.expand_dims(tar[:,t],1)
	batch_loss = (loss/int(tar.shape[1]))
	variables = encoder.trainable_variables+decoder.trainable_variables
	gradients = tape.gradient(loss,variables)
	optimizer.apply_gradients(zip(gradients,variables))
	return batch_loss

for epoch in range(10):
	enc_hidden = encoder.initialize_hidden_state()
	for batch,(inp,tar) in enumerate(dataset.take(steps_per_epoch)):
		batch_loss = train_step(inp,tar,enc_hidden)
		if (batch+1)%100 == 0:
			print(epoch+1,batch,batch_loss.numpy())
	if (epoch+1)%2 == 0:
		checkpoint.save(file_prefix=checkpoint_prefix)

def evaluate(sentence):
	# attention_plot = np.zeros((max_length_tar,max_length_inp))
	sentence = preprocess_sentence(sentence)
	inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
	inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=max_length_inp,padding='post')
	inputs = tf.convert_to_tensor(inputs)
	result = ''; hidden = [tf.zeros((1,units))]
	enc_out,enc_hidden = encoder(inputs,hidden)
	dec_hidden = enc_hidden
	dec_input = tf.expand_dims([tar_lang.word_index['<start>']],0)
	for t in range(max_length_tar):
		predictions,dec_hidden,attention_weights = decoder(dec_input,dec_hidden,enc_out)
		attention_weights = tf.reshape(attention_weights,(-1,))
		# attention_plot[t] = attention_weights.numpy()
		predicted_id = tf.argmax(predictions[0]).numpy()
		result += tar_lang.index_word[predicted_id]+' '
		if tar_lang.index_word[predicted_id] == '<end>':
			return result, sentence # attention_plot
		dec_input = tf.expand_dims([predicted_id],0)
	return result, sentence # attention_plot

# def plot_attention(attention, sentence, predicted_sentence):
# 	fig = plt.figure(figsize=(10,10))
# 	ax = fig.add_subplot(1, 1, 1)
# 	ax.matshow(attention,cmap='viridis')
# 	fontdict = {'fontsize':14}
# 	ax.set_xticklabels(['']+sentence,fontdict=fontdict,rotation=90)
# 	ax.set_yticklabels(['']+predicted_sentence,fontdict=fontdict)
# 	plt.show()

def translate(sentence):
	result,sentence = evaluate(sentence) # attention_plot
	print(sentence,'\n',result)
	# attention_plot = attention_plot[:len(result.split(' ')),:len(sentence.split(' '))]
	# plot_attention(attention_plot,sentence.split(' '),result.split(' '))

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
translate(u'hace mucho frio aqui.')
