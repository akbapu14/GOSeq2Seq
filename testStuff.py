import tensorflow as tf
from seq2seq.models import bridges
from seq2seq.encoders import rnn_encoder
from seq2seq.decoders import attention_decoder
from seq2seq.decoders import basic_decoder
from seq2seq.decoders import attention
from seq2seq.contrib.seq2seq import helper as tf_decode_helper
from tensorflow.contrib.rnn import LSTMStateTuple
#Parameters
input_vocab_size = 20
output_vocab_size = 50
input_embedding_size = 20
output_embedding_size = 50
#Inputs

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_lengths')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

#Embeddings
input_embeddings = tf.Variable(tf.random_uniform([input_vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
output_embeddings = tf.Variable(tf.random_uniform([output_embedding_size, output_vocab_size], -1.0, 1.0), dtype=tf.float32)

#Network
encoder_inputs_embedded = tf.nn.embedding_lookup(input_embeddings, encoder_inputs, name="input_embedding_vector")
decoder_targets_embedded = tf.nn.embedding_lookup(output_embeddings, decoder_targets, name="decoder_embedding_vector")

encoder = rnn_encoder.BidirectionalRNNEncoder(params={}, mode=tf.contrib.learn.ModeKeys.TRAIN)

eout = encoder.encode(encoder_inputs_embedded, encoder_inputs_length)
encoder_logits = tf.contrib.layers.linear(eout.outputs, output_embedding_size)
prediction = tf.argmax(encoder_logits, 2)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=output_vocab_size, dtype=tf.float32),
    logits=encoder_logits,
)
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init_op = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
sess.run(init_op)
sess.run(init_l)

# testArray = [[[1,2,3,4,5], [1,2,3,0,0]], [[1,2,3,0,0], [0,0,0,0,0]]]
testArray = [[1,2,3,4,5], [1,2,3,0,0]]
valArray = [[6,5,4,3,2],[6,5,4,3,2]]
endArray = [[30,20,10,0,0], [10,4,35,0,0]]
for i in range(1000):
    print(sess.run([train_op, loss, prediction], {encoder_inputs: testArray, encoder_inputs_length: [5,5], decoder_targets: endArray}))
# print(sess.run(prediction, {encoder_inputs: valArray, encoder_inputs_length: [5,5], decoder_targets: endArray}))
