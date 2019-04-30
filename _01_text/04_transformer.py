# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

examples,metadata = tfds.load('ted_hrlr_translate/pt_to_en',with_info=True,as_supervised=True)
train_examples,val_examples = examples['train'],examples['validation']
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples),target_vocab_size=2**13)
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples),target_vocab_size=2**13)

BUFFER_SIZE = 20000; BATCH_SIZE = 64; MAX_LENGTH = 40

def encode(lang1, lang2):
	lang1 = [tokenizer_pt.vocab_size]+tokenizer_pt.encode(lang1.numpy())+[tokenizer_pt.vocab_size+1]
	lang2 = [tokenizer_en.vocab_size]+tokenizer_en.encode(lang2.numpy())+[tokenizer_en.vocab_size+1]
	return lang1, lang2

def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x)<=max_length,tf.size(y)<=max_length)

train_dataset = train_examples.map(lambda pt,en:tf.py_function(encode,[pt,en],[tf.int64,tf.int64]))
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,padded_shapes=([-1],[-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_examples.map(lambda pt,en:tf.py_function(encode,[pt,en],[tf.int64,tf.int64]))
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE,padded_shapes=([-1],[-1]))

def get_angles(pos, i, d_model):
	angle_rates = 1/np.power(10000,(2*(i//2))/np.float32(d_model))
	return pos*angle_rates

def positional_encoding(position, d_model):
	angle_rads = get_angles(np.arange(position)[:,np.newaxis],np.arange(d_model)[np.newaxis,:],d_model)
	sins = np.sin(angle_rads[:,0::2])
	coss = np.cos(angle_rads[:,1::2])
	pos_encoding = np.concatenate([sins,coss],axis=-1)[np.newaxis,...]
	return tf.cast(pos_encoding,dtype=tf.float32)

def create_padding_mask(seq):
	seq = tf.cast(tf.math.equal(seq,0),tf.float32)
	return seq[:,tf.newaxis,tf.newaxis,:] # (batch_size,1,1,seq_len)

def create_look_ahead_mask(size):
	mask = 1-tf.linalg.band_part(tf.ones((size,size)),-1,0)
	return mask  # (seq_len,seq_len)

def scaled_dot_product_attention(q, k, v, mask):
	matmul_qk = tf.matmul(q,k,transpose_b=True)
	dk = tf.cast(tf.shape(k)[-1],tf.float32)
	scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)
	if mask is not None:
		scaled_attention_logits += (mask*-1e9)
	attention_weights = tf.nn.softmax(scaled_attention_logits,axis=-1)
	output = tf.matmul(attention_weights,v)
	return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention,self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model
		assert d_model%self.num_heads == 0
		self.depth = d_model//self.num_heads
		self.wq = tf.keras.layers.Dense(d_model)
		self.wk = tf.keras.layers.Dense(d_model)
		self.wv = tf.keras.layers.Dense(d_model)
		self.dense = tf.keras.layers.Dense(d_model)
	def split_heads(self, x, batch_size):
		x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
		return tf.transpose(x,perm=[0,2,1,3])
	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]
		q = self.wq(q)
		k = self.wk(k)
		v = self.wv(v)
		q = self.split_heads(q,batch_size)
		k = self.split_heads(k,batch_size)
		v = self.split_heads(v,batch_size)
		scaled_attention,attention_weights = scaled_dot_product_attention(q,k,v,mask)
		scaled_attention = tf.transpose(scaled_attention,perm=[0,2,1,3])
		concat_attention = tf.reshape(scaled_attention,(batch_size,-1,self.d_model))
		output = self.dense(concat_attention)
		return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
	return tf.keras.Sequential([
		tf.keras.layers.Dense(dff,activation='relu'),
		tf.keras.layers.Dense(d_model)])

class EncoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(EncoderLayer,self).__init__()
		self.mha = MultiHeadAttention(d_model,num_heads)
		self.ffn = point_wise_feed_forward_network(d_model,dff)
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)
	def call(self, x, training, mask):
		attn_output,_ = self.mha(x,x,x,mask)
		attn_output = self.dropout1(attn_output,training=training)
		out1 = self.layernorm1(x+attn_output)
		ffn_output = self.ffn(out1)
		ffn_output = self.dropout2(ffn_output,training=training)
		out2 = self.layernorm2(out1+ffn_output)
		return out2

class DecoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(DecoderLayer,self).__init__()
		self.mha1 = MultiHeadAttention(d_model,num_heads)
		self.mha2 = MultiHeadAttention(d_model,num_heads)
		self.ffn = point_wise_feed_forward_network(d_model,dff)
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)
		self.dropout3 = tf.keras.layers.Dropout(rate)
	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
		attn1,attn_weights_block1 = self.mha1(x,x,x,look_ahead_mask)
		attn1 = self.dropout1(attn1,training=training)
		out1 = self.layernorm1(attn1+x)
		attn2,attn_weights_block2 = self.mha2(enc_output,enc_output,out1,padding_mask)
		attn2 = self.dropout2(attn2,training=training)
		out2 = self.layernorm2(attn2+out1)
		ffn_output = self.ffn(out2)
		ffn_output = self.dropout3(ffn_output,training=training)
		out3 = self.layernorm3(ffn_output+out2)
		return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
		super(Encoder,self).__init__()
		self.d_model = d_model
		self.num_layers = num_layers
		self.embedding = tf.keras.layers.Embedding(input_vocab_size,d_model)
		self.pos_encoding = positional_encoding(input_vocab_size,self.d_model)
		self.enc_layers = [EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]
		self.dropout = tf.keras.layers.Dropout(rate)
	def call(self, x, training, mask):
		seq_len = tf.shape(x)[1]
		x = self.embedding(x)
		x *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
		x += self.pos_encoding[:,:seq_len,:]
		x = self.dropout(x,training=training)
		for i in range(self.num_layers):
			x = self.enc_layers[i](x,training, mask)
		return x

class Decoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
		super(Decoder,self).__init__()
		self.d_model = d_model
		self.num_layers = num_layers
		self.embedding = tf.keras.layers.Embedding(target_vocab_size,d_model)
		self.pos_encoding = positional_encoding(target_vocab_size,self.d_model)
		self.dec_layers = [DecoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]
		self.dropout = tf.keras.layers.Dropout(rate)
	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
		seq_len = tf.shape(x)[1]
		attention_weights = {}
		x = self.embedding(x)
		x *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
		x += self.pos_encoding[:,:seq_len,:]
		x = self.dropout(x,training=training)
		for i in range(self.num_layers):
			x,block1,block2 = self.dec_layers[i](x,enc_output,training,look_ahead_mask,padding_mask)
			attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
			attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
		return x, attention_weights

class Transformer(tf.keras.Model):
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, rate=0.1):
		super(Transformer, self).__init__()
		self.encoder = Encoder(num_layers,d_model,num_heads,dff,input_vocab_size,rate)
		self.decoder = Decoder(num_layers,d_model,num_heads,dff,target_vocab_size,rate)
		self.final_layer = tf.keras.layers.Dense(target_vocab_size)
	def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
		enc_output = self.encoder(inp,training,enc_padding_mask)
		dec_output, attention_weights = self.decoder(tar,enc_output,training,look_ahead_mask,dec_padding_mask)
		final_output = self.final_layer(dec_output)
		return final_output, attention_weights

num_layers = 4; d_model = 128; dff = 512; num_heads = 8; dropout_rate = 0.1
input_vocab_size = tokenizer_pt.vocab_size+2
target_vocab_size = tokenizer_en.vocab_size+2

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=4000):
		super(CustomSchedule,self).__init__()
		self.d_model = d_model
		self.d_model = tf.cast(self.d_model,tf.float32)
		self.warmup_steps = warmup_steps
	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step*(self.warmup_steps**-1.5)
		return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1,arg2)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate,beta_1=0.9,beta_2=0.98,epsilon=1e-9)

def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real,0))
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')(real,pred)
	return tf.reduce_mean(loss*tf.cast(mask,dtype=loss.dtype))

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

transformer = Transformer(num_layers,d_model,num_heads,dff,input_vocab_size,target_vocab_size,dropout_rate)

def create_masks(inp, tar):
	enc_padding_mask = create_padding_mask(inp)
	dec_padding_mask = create_padding_mask(inp)
	look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
	dec_target_padding_mask = create_padding_mask(tar)
	combined_mask = tf.maximum(dec_target_padding_mask,look_ahead_mask)
	return enc_padding_mask, combined_mask, dec_padding_mask

checkpoint_path = './checkpoints/train'
ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt,checkpoint_path,max_to_keep=5)
if ckpt_manager.latest_checkpoint: ckpt.restore(ckpt_manager.latest_checkpoint)

@tf.function
def train_step(inp, tar):
	tar_inp,tar_real = tar[:,:-1],tar[:,1:]
	enc_padding_mask,combined_mask,dec_padding_mask = create_masks(inp,tar_inp)
	with tf.GradientTape() as tape:
		predictions, _ = transformer(inp,tar_inp,True,enc_padding_mask,combined_mask,dec_padding_mask)
		loss = loss_function(tar_real,predictions)
	gradients = tape.gradient(loss,transformer.trainable_variables)    
	optimizer.apply_gradients(zip(gradients,transformer.trainable_variables))
	train_loss(loss); train_accuracy(tar_real,predictions)

for epoch in range(20):  
	train_loss.reset_states()
	train_accuracy.reset_states()
	for batch,(inp,tar) in enumerate(train_dataset):
		train_step(inp,tar)
		if (batch+1)%500 == 0:
			print(epoch+1,batch,train_loss.result(),train_accuracy.result())
	if (epoch+1)%5 == 0:
		ckpt_save_path = ckpt_manager.save()

def evaluate(inp_sentence):
	start_token,end_token = [tokenizer_pt.vocab_size],[tokenizer_pt.vocab_size+1]
	inp_sentence = start_token+tokenizer_pt.encode(inp_sentence)+end_token
	encoder_input = tf.expand_dims(inp_sentence,0)
	decoder_input = [tokenizer_en.vocab_size]
	output = tf.expand_dims(decoder_input,0)
	for i in range(MAX_LENGTH):
		enc_padding_mask,combined_mask,dec_padding_mask = create_masks(encoder_input,output)
		predictions,attention_weights = transformer(encoder_input,output,False,enc_padding_mask,combined_mask,dec_padding_mask)
		predictions = predictions[:,-1:,:]
		predicted_id = tf.cast(tf.argmax(predictions,axis=-1),tf.int32)
		if tf.equal(predicted_id,tokenizer_en.vocab_size+1):
			return tf.squeeze(output,axis=0), attention_weights
	output = tf.concat([output,predicted_id],axis=-1)
	return tf.squeeze(output,axis=0), attention_weights

def translate(sentence):
	result, attention_weights = evaluate(sentence)
	predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])  
	print(sentence,'\n',predicted_sentence)

translate('este é um problema que temos que resolver.')
