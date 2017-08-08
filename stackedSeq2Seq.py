import tensorflow as tf
from seq2seq.models import bridges
from seq2seq.encoders import rnn_encoder
from seq2seq.decoders import attention_decoder
from seq2seq.decoders import basic_decoder
from seq2seq.decoders import attention
from seq2seq.contrib.seq2seq import helper as tf_decode_helper
from tensorflow.contrib.rnn import LSTMStateTuple
import numpy as np
#Parameters
input_vocab_size = 20
output_vocab_size = 100
input_embedding_size = 20
output_embedding_size = 50
numberArticles = 2
#Inputs
pmode = tf.contrib.learn.ModeKeys.INFER
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_lengths')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
lengthOfArticles= tf.placeholder(shape=(None,), dtype=tf.int32, name='l')

#Embeddings
input_embeddings = tf.Variable(tf.random_uniform([input_vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
output_embeddings = tf.Variable(tf.random_uniform([output_vocab_size, output_embedding_size], -1.0, 1.0), dtype=tf.float32)

#Network
encoder_inputs_embedded = tf.nn.embedding_lookup(input_embeddings, encoder_inputs, name="input_embedding_vector")
decoder_targets_embedded = tf.nn.embedding_lookup(output_embeddings, decoder_targets, name="decoder_embedding_vector")

encoder = rnn_encoder.BidirectionalRNNEncoder(params={}, mode=pmode)

eout = encoder.encode(encoder_inputs_embedded, encoder_inputs_length)
# eout2 = encoder.encode(tf.nn.embedding_lookup(input_embeddings, article_inputs[1]), encoder_inputs_length)

#eout.attention_values = (4,5,256)
#eout.attention_values_length = [5,5,5,5]
#eout.outputs = (4,5,256)
#eout.final_state -> encoder_final_state -> LSTMStateTuple[0] = (4,128)

def sumUp(someTensor, numValues):
    sumOver = tf.add_n([tf.multiply(someTensor[0], 1/float(numValues)), tf.multiply(someTensor[1], 1/float(numValues))])
    sumOver2 = tf.add_n([tf.multiply(someTensor[0], 1/float(numValues)), tf.multiply(someTensor[1], 1/float(numValues))])
    # currentArticle = 0
    # #Need to separate by lengthofarticles ([2,2])
    # sumOver.append([])
    # def newA():
    #     nonlocal currentArticle
    #     currentArticle += 1
    #     sumOver.append([])
    #     return sameA()
    # def sameA():
    #     sumOver[currentArticle].append(tf.multiply(someTensor[i], 1/float(numValues)))
    #     return tf.constant(5)
    # for i in range(numValues):
    #     _ = tf.cond(i > lengthOfArticles[currentArticle], newA, sameA)
    return tf.stack([sumOver, sumOver2])


summedAttention = sumUp(eout.attention_values, 4)
summedLengths = eout.attention_values_length[:1]
summedOutputs = sumUp(eout.outputs, 4)

# ***
decoder = attention_decoder.AttentionDecoder(params={}, mode=pmode,
vocab_size=output_vocab_size,
attention_values=eout.attention_values,
attention_values_length=eout.attention_values_length,
attention_keys=eout.outputs,
attention_fn=attention.AttentionLayerBahdanau(params={}, mode=pmode))

# decoder = attention_decoder.AttentionDecoder(params={}, mode=tf.contrib.learn.ModeKeys.TRAIN,
# vocab_size=output_vocab_size,
# attention_values=summedAttention,
# attention_values_length=summedLengths,
# attention_keys=summedOutputs,
# attention_fn=attention.AttentionLayerBahdanau(params={}, mode=tf.contrib.learn.ModeKeys.TRAIN))
# decoder = basic_decoder.BasicDecoder(params={}, mode=tf.contrib.learn.ModeKeys.TRAIN, vocab_size=output_vocab_size)



# bridge = bridges.InitialStateBridge(encoder_outputs=eout,
# decoder_state_size=128,
# params={},
# mode=tf.contrib.learn.ModeKeys.TRAIN)
batch_size = 4
target_start_id = 1
helper_infer = tf_decode_helper.GreedyEmbeddingHelper(
    embedding=output_embeddings,
    start_tokens=tf.fill([batch_size], target_start_id),
    end_token=5)
# helper_train = tf_decode_helper.TrainingHelper(
#         inputs=decoder_targets_embedded[:, :-1],
#         sequence_length=decoder_targets_length - 1)
# decoder_initial_state = bridge()
# print(decoder_initial_state)
dstate = eout.final_state

encoder_final_state_c = tf.add(tf.multiply(dstate[0].c, .5), tf.multiply(dstate[1].c, .5))
encoder_final_state_h = tf.add(tf.multiply(dstate[0].h, .5), tf.multiply(dstate[1].h, .5))

# summed_encoder_final_state_c = tf.add(tf.multiply(sumUp(dstate[0].c, 4), .5), tf.multiply(sumUp(dstate[1].c, 4), .5))
# summed_encoder_final_state_h = tf.add(tf.multiply(sumUp(dstate[0].h, 4), .5), tf.multiply(sumUp(dstate[1].h, 4), .5))
# encoder_final_state_h = tf.concat(
#     (dstate[0].h, dstate[1].h), 1)
#
encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)
# summed_encoder_final_state = LSTMStateTuple(
#     c=summed_encoder_final_state_c,
#     h=summed_encoder_final_state_h
# )
# dstate2 = eout2.final_state
# summed_encoder_final_state = tf.Print(summed_encoder_final_state, [1.0, 3.0], message="On to decoding")

dout, _, = decoder(encoder_final_state, helper_infer)


sess = tf.Session()
init_op = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
sess.run(init_op)
sess.run(init_l)

testArray = [[0,0,0,0,0], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]]
# valArray = [[6,5,4,3,2],[6,5,4,3,2]]
endArray = [[1,2,3,4,0], [1,2,3,4,0], [1,2,3,4,0], [0,0,0,0,0]]

d = sess.run(dout, {encoder_inputs: testArray, encoder_inputs_length: [5,5,5,5], lengthOfArticles: [2,2]})


# dout = decoder.decode(eout.outputs, encoder_inputs_embedded, decoder_targets)
# encoder_logits = tf.contrib.layers.linear(outputs, output_embedding_size)
# decoder_logits = tf.contrib.layers.linear(encoder_logits, output_vocab_size)
# prediction = tf.argmax(decoder_logits, 2)


#Training
# stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
#     labels=tf.one_hot(decoder_targets, depth=output_vocab_size, dtype=tf.float32),
#     logits=decoder_logits,
# )
# loss = tf.reduce_mean(stepwise_cross_entropy)
# train_op = tf.train.AdamOptimizer().minimize(loss)

# decoder_targets_predicted = tf.nn.embedding_lookup(output_embeddings, decoder_targets)

# for i in range(1000):
#     print(sess.run([train_op, loss, prediction], {encoder_inputs: testArray, encoder_inputs_length: [5,5], decoder_targets: endArray}))
# print(sess.run(prediction, {encoder_inputs: valArray, encoder_inputs_length: [5,5], decoder_targets: endArray}))
# test = tf.get_default_graph().get_tensor_by_name("first_inputs")
# with tf.device("/gpu:0"):
#     encoder2 = rnn_encoder.UnidirectionalRNNEncoder(params={}, mode=tf.contrib.learn.ModeKeys.TRAIN)
#     eout2 = encoder2.encode(encoder_inputs_embedded, encoder_inputs_length)
#
#
# with tf.device("/gpu:0"):
#     encoder3 = rnn_encoder.StackBidirectionalRNNEncoder(params={}, mode=tf.contrib.learn.ModeKeys.TRAIN)
#     eout3 = encoder3.encode(encoder_inputs_embedded, encoder_inputs_length)
