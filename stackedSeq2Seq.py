import random
import tensorflow as tf
from seq2seq.models import bridges
from seq2seq.encoders import rnn_encoder
from seq2seq.decoders import attention_decoder
from seq2seq.decoders import basic_decoder
from seq2seq.decoders import attention
from seq2seq.training import utils as training_utils
from seq2seq.contrib.seq2seq.decoder import _transpose_batch_time
from seq2seq.contrib.seq2seq import helper as tf_decode_helper
from tensorflow.contrib.rnn import LSTMStateTuple
from seq2seq import losses as seq2seq_losses
import collections
from seq2seq.models.model_base import ModelBase, _flatten_dict
import numpy as np
import time
import pickle




#Parameters
input_vocab_size = 96100 + 5
output_vocab_size = 96582 + 3
input_embedding_size = 500
output_embedding_size = 500
numberArticles = 10

optimizer_params = {
    "optimizer.name": "Adam",
    "optimizer.learning_rate": 1e-4,
    "optimizer.params": {}, # Arbitrary parameters for the optimizer
    "optimizer.lr_decay_type": "",
    "optimizer.lr_decay_steps": 100,
    "optimizer.lr_decay_rate": 0.99,
    "optimizer.lr_start_decay_at": 0,
    "optimizer.lr_stop_decay_at": tf.int32.max,
    "optimizer.lr_min_learning_rate": 1e-12,
    "optimizer.lr_staircase": False,
    "optimizer.clip_gradients": 5.0,
    "optimizer.sync_replicas": 0,
    "optimizer.sync_replicas_to_aggregate": 0,
}

#Akilesh's version of tackling articles with multiple sentences by breaking up and summing attention/outputs

#Inputs
pmode = tf.contrib.learn.ModeKeys.TRAIN
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_lengths')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
articleIndicators= tf.placeholder(shape=(None,), dtype=tf.int32, name='l')
# numValues = tf.placeholder(tf.float32)
#Embeddings
input_embeddings = tf.Variable(tf.random_uniform([input_vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
output_embeddings = tf.Variable(tf.random_uniform([output_vocab_size, output_embedding_size], -1.0, 1.0), dtype=tf.float32)

#Network
encoder_inputs_embedded = tf.nn.embedding_lookup(input_embeddings, encoder_inputs, name="input_embedding_vector")
decoder_targets_embedded = tf.nn.embedding_lookup(output_embeddings, decoder_targets, name="decoder_embedding_vector")

encoder = rnn_encoder.BidirectionalRNNEncoder(params={}, mode=pmode)

eout = encoder.encode(encoder_inputs_embedded, encoder_inputs_length)

#eout.attention_values = (4,5,256)
#eout.attention_values_length = [5,5,5,5]
#eout.outputs = (4,5,256)
#eout.final_state -> encoder_final_state -> LSTMStateTuple[0] = (4,128)
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
def sumUp(someTensor):
    #Tensors where first dimension is what you need to concatenate over
    # weightedTensor = tf.Print(weightedTensor, tf.divide(1, numValues))
    partitioned = tf.dynamic_partition(data=someTensor, partitions=articleIndicators, num_partitions=numberArticles, name="Partition_Data")
    finalList = [tf.reduce_mean(tensor, axis=0) for tensor in partitioned]
    return tf.stack(finalList)
def compute_loss(decoder_output, labels, labelLengths):
    """Computes the loss for this model.

    Returns a tuple `(losses, loss)`, where `losses` are the per-batch
    losses and loss is a single scalar tensor to minimize.
    """
    #pylint: disable=R0201
    # Calculate loss per example-timestep of shape [B, T]
    losses = seq2seq_losses.cross_entropy_sequence_loss(
        logits=decoder_output.logits[:, :, :],
        targets=tf.transpose(labels[:, 1:], [1, 0]),
        sequence_length=labelLengths - 1)

    # Calculate the average log perplexity
    loss = tf.reduce_sum(losses) / tf.to_float(
        tf.reduce_sum(labelLengths - 1))

    return losses, loss
def _create_optimizer():
    """Creates the optimizer"""
    name = optimizer_params["optimizer.name"]
    optimizer = tf.contrib.layers.OPTIMIZER_CLS_NAMES[name](
        learning_rate=optimizer_params["optimizer.learning_rate"],
        **optimizer_params["optimizer.params"])
    return optimizer
def _clip_gradients(grads_and_vars):
    """Clips gradients by global norm."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, optimizer_params["optimizer.clip_gradients"])
    return list(zip(clipped_gradients, variables))
def _build_train_op(loss):
    """Creates the training operation"""
    learning_rate_decay_fn = training_utils.create_learning_rate_decay_fn(
        decay_type=optimizer_params["optimizer.lr_decay_type"] or None,
        decay_steps=optimizer_params["optimizer.lr_decay_steps"],
        decay_rate=optimizer_params["optimizer.lr_decay_rate"],
        start_decay_at=optimizer_params["optimizer.lr_start_decay_at"],
        stop_decay_at=optimizer_params["optimizer.lr_stop_decay_at"],
        min_learning_rate=optimizer_params["optimizer.lr_min_learning_rate"],
        staircase=optimizer_params["optimizer.lr_staircase"])

    optimizer = _create_optimizer()
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=optimizer_params["optimizer.learning_rate"],
        learning_rate_decay_fn=learning_rate_decay_fn,
        clip_gradients=_clip_gradients,
        optimizer=optimizer,
        summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

    return train_op

def predict(decoder_output):
    predictions = {}
    # Decoders returns output in time-major form [T, B, ...]
    # Here we transpose everything back to batch-major for the user
    output_dict = collections.OrderedDict(
        zip(decoder_output._fields, decoder_output))
    decoder_output_flat = _flatten_dict(output_dict)
    decoder_output_flat = {
        k: _transpose_batch_time(v)
        for k, v in decoder_output_flat.items()
    }
    predictions.update(decoder_output_flat)
    return predictions

def hbatch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
#    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_batch_major, sequence_lengths

summedAttention = sumUp(eout.attention_values)
summedLengths = eout.attention_values_length[:1]
summedOutputs = sumUp(eout.outputs)

decoder = attention_decoder.AttentionDecoder(params={}, mode=pmode,
vocab_size=output_vocab_size,
attention_values=summedAttention,
attention_values_length=summedLengths,
attention_keys=summedOutputs,
attention_fn=attention.AttentionLayerBahdanau(params={}, mode=pmode))

batch_size = 2
target_start_id = 1
# helper_infer = tf_decode_helper.GreedyEmbeddingHelper(
#     embedding=output_embeddings,
#     start_tokens=tf.fill([batch_size], target_start_id),
#     end_token=5)
helper_train = tf_decode_helper.TrainingHelper(
        inputs=decoder_targets_embedded[:, :-1],
        sequence_length=decoder_targets_length - 1)
dstate = eout.final_state

summed_encoder_final_state_c = tf.add(tf.multiply(sumUp(dstate[0].c), .5), tf.multiply(sumUp(dstate[1].c), .5))
summed_encoder_final_state_h = tf.add(tf.multiply(sumUp(dstate[0].h), .5), tf.multiply(sumUp(dstate[1].h), .5))
summed_encoder_final_state = LSTMStateTuple(
    c=summed_encoder_final_state_c,
    h=summed_encoder_final_state_h
)

decoder_output, _, = decoder(summed_encoder_final_state, helper_train)
#

predictions = predict(decoder_output)['predicted_ids']
losses, loss = compute_loss(decoder_output=decoder_output, labels=decoder_targets, labelLengths=decoder_targets_length)
train_op = _build_train_op(loss)


sess = tf.Session()
init_op = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
saver = tf.train.Saver()
sess.run(init_op)
sess.run(init_l)

testArray = [[1,2,3,4,5], [6,7,8,9,10], [1,2,3,6,5], [1,2,3,4,5], [1,2,3,4,5], [6,7,8,9,10], [1,2,3,6,5], [1,2,3,4,5]]
valArray = [[6,5,4,3,2] * 20,[7,5,4,34,2] * 20, [7,5,4,34,2] * 20, [7,5,4,34,2] * 20,[7,5,4,34,2] * 20]
test1Array = [[1,2,3,4,5], [6,7,8,9,10], [1,2,3,6,5], [1,2,3,4,5], [1,2,3,4,5], [6,7,8,9,10], [1,2,3,6,5], [1,2,3,4,5], [1,2,3,4,5]]
endArray = [[6,5,4,3,2] * 20,[7,5,4,34,2] * 20, [7,5,4,34,2] * 20, [7,5,4,34,2] * 20,[7,5,4,34,2] * 20]
stacked_articles_train = load_obj("../stacked_articles_train")
stacked_annotations_train = load_obj("../stacked_annotations_train")
def getNext():
    list_of_random_indices = random.sample(list(range(len(stacked_articles_train))), numberArticles)
    print(list_of_random_indices)
    inputs = []
    targets = []
    articleIndicators = []
    for i,index in enumerate(list_of_random_indices):
        for l in stacked_articles_train[index]:
            inputs.append(l)
            articleIndicators.append(i)
        targets.append(stacked_annotations_train[index])
    features, features_lengths = hbatch(inputs)
    labels, labels_lengths = hbatch(targets)
    # features = [[1,2,3,4,5], [6,7,8,9,10], [1,2,3,6,5], [1,2,3,4,5], [1,2,3,4,5], [6,7,8,9,10], [1,2,3,6,5], [1,2,3,4,5]]
    # labels = [[6,5,4,3,2] * 20,[7,5,4,34,2] * 20, [7,5,4,34,2] * 20, [7,5,4,34,2] * 20,[7,5,4,34,2] * 20]
    # features_lengths = [5] * 8
    # labels_lengths = [100]  * 5
    # articleIndicators = [0,0,1,1,2,3,4,4]

    return [features, labels, features_lengths, labels_lengths, articleIndicators]
#Training Cycle

for i in range(10000):
    start = time.time()
    # saver.restore(sess, "model.ckpt")
    # print("Model restored.")
    
    inputs = getNext()
    while len(inputs[4]) < 750:
        inputs = getNext()
    print(sess.run([train_op], {encoder_inputs: inputs[0], decoder_targets: inputs[1], encoder_inputs_length: inputs[2], decoder_targets_length: inputs[3], articleIndicators: inputs[4]}))
    # print(sess.run([train_op], {encoder_inputs: test1Array, encoder_inputs_length: [5] * 9, lengthOfArticles: [0,0,1,1,2,3,4,4,3], decoder_targets: endArray, decoder_targets_length: [100]  * 5}))

    print("This batch of 10 took: " + str(time.time() - start) + " seconds")
    # save_path = saver.save(sess, "model.ckpt")
    # print("Model saved in file: %s" % save_path)
# d = sess.run([train_op], {encoder_inputs: testArray, encoder_inputs_length: [5] * 8, lengthOfArticles: [0,0,1,1,2,3,4,4], decoder_targets: valArray, decoder_targets_length: [100]  * 5})















# bridge = bridges.InitialStateBridge(encoder_outputs=eout,
# decoder_state_size=128,
# params={},
# mode=tf.contrib.learn.ModeKeys.TRAIN)
# decoder_initial_state = bridge()

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
