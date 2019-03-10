import tensorflow as tf
import logging
import sys
from config import FLAGS
from utils import build_vocab
from functools import partial
import numpy as np

# Logging
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers

def parse_function(mode, example_proto):
    '''Parse example protobuf
    Args:
        mode: The mode can be 'train', 'predict' and 'evaluation', indicate the running mode
        example_proto: example protobuf which is read from a .tfrecords file
    Returns:
        example: a dict, the key of the dict is the feature name, the value is a tensor
    '''

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        features = {
            'context': tf.VarLenFeature(dtype=tf.int64),
            'utterance': tf.VarLenFeature(dtype=tf.int64),
            'context_length': tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'utterance_length': tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'label': tf.VarLenFeature(dtype=tf.float32)
        }
        example = tf.parse_single_example(example_proto, features)
        example['context'] = tf.sparse_tensor_to_dense(example['context'])
        example['utterance'] = tf.sparse_tensor_to_dense(example['utterance'])
        example['label'] = tf.sparse_tensor_to_dense(example['label'])
    else:
        features = {
            'context': tf.VarLenFeature(dtype=tf.int64),
            'utterance': tf.VarLenFeature(dtype=tf.int64),
            'context_length': tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'utterance_length': tf.FixedLenFeature(shape=[], dtype=tf.int64)
        }
        for i in range(9):
            features['dis_{}'.format(i)] = tf.VarLenFeature(dtype=tf.int64)
            features['dis_{}_length'.format(i)] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
        example = tf.parse_single_example(example_proto, features)
        example['context'] = tf.sparse_tensor_to_dense(example['context'])
        example['utterance'] = tf.sparse_tensor_to_dense(example['utterance'])
        for i in range(9):
            example['dis_{}'.format(i)] = tf.sparse_tensor_to_dense(example['dis_{}'.format(i)])

    # tf.Example only supports tf.int64, so we need to cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def create_input_fn(fname, params, mode, shuffle_and_repeat=False):
    '''Create input function for training, testing and evaluating'''
    def input_fn():
        dataset = tf.data.TFRecordDataset([fname])
        dataset = dataset.map(partial(parse_function, mode))
        if mode == tf.estimator.ModeKeys.PREDICT:
            padded_shapes = {
                'context': [None],
                'utterance': [None],
                'context_length': (),
                'utterance_length': ()
            }
            for i in range(9):
                padded_shapes['dis_{}'.format(i)] = [None]
                padded_shapes['dis_{}_length'.format(i)] = ()
            dataset = dataset.padded_batch(params.get('batch_size', 1), padded_shapes=padded_shapes)
        else:
            dataset = dataset.padded_batch(params.get('batch_size', 1), padded_shapes={'context': [None], 'utterance': [None], 'context_length': (), 'utterance_length': (), 'label': [None]})
        if shuffle_and_repeat:
            dataset = dataset.shuffle(buffer_size=params['buffer_size']).repeat(params['num_epochs'])
        return dataset

    return input_fn


def model_fn(features, labels, mode, params):
    '''We use a dual lstm to capture the relationship between context and utterance, and multiply it with a matrix,
    at last, we use a sigmoid function to predict the probability of utterance being the ground truth response.
    for more details refer to the paper https://arxiv.org/abs/1506.08909
    '''
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        context, utterance, context_length, utterance_length, labels = features['context'], features['utterance'], features['context_length'], features['utterance_length'], features['label']
    else:
        context, utterance, context_length, utterance_length = features['context'], features['utterance'], features['context_length'], features['utterance_length']
        all_contexts, all_utterances = [context], [utterance]
        all_contexts_length, all_utterances_length = [context_length], [utterance_length]
        all_targets = [tf.ones(1, dtype=tf.float32)]
        for i in range(9):
            distractor, distractor_length = features['dis_{}'.format(i)], features['dis_{}_length'.format(i)]
            all_contexts.append(context)
            all_contexts_length.append(context_length)
            all_utterances.append(distractor)
            all_utterances_length.append(distractor_length)
            all_targets.append(tf.zeros(1, dtype=tf.float32))
        context = tf.concat(all_contexts, axis=0)
        utterance = tf.concat(all_utterances, axis=0)
        context_length = tf.concat(all_contexts_length, axis=0)
        utterance_length = tf.concat(all_utterances_length, axis=0)
        labels = tf.concat(all_targets, axis=0)

    embedding_matrix = tf.Variable(tf.random_normal(shape=[params['vocab_size'], FLAGS.embedding_size], stddev=0.01), name='embedding_matrix')
    M = tf.Variable(tf.random_normal(shape=[FLAGS.hidden_size, FLAGS.hidden_size]), name='M')
    context_embedded = tf.nn.embedding_lookup(embedding_matrix, context, name='context_embedded')
    utterance_embedded = tf.nn.embedding_lookup(embedding_matrix, utterance, name='utterance_embedded')

    context_embedded = tf.transpose(context_embedded, perm=[1, 0, 2]) # time major in LSTMBlockFusedCell
    utterance_embedded = tf.transpose(utterance_embedded, perm=[1, 0, 2])

    with tf.variable_scope('context') as vc:
        context_lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(FLAGS.hidden_size, name='context_lstm_cell')
        _, context_state = context_lstm_cell(context_embedded, sequence_length=context_length, dtype=tf.float32)
        # context_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.hidden_size, initializer=tf.orthogonal_initializer)
        # _, context_state = tf.nn.dynamic_rnn(context_lstm_cell, context_embedded, sequence_length=context_length, dtype=tf.float32)
    with tf.variable_scope('utterance') as vu:
        utterance_lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(FLAGS.hidden_size, name='utterance_lstm_cell')
        _, utterance_state = utterance_lstm_cell(utterance_embedded, sequence_length=utterance_length, dtype=tf.float32)
        # utterance_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.hidden_size, initializer=tf.orthogonal_initializer)
        # _, utterance_state = tf.nn.dynamic_rnn(utterance_lstm_cell, utterance_embedded, sequence_length=utterance_length, dtype=tf.float32)
    context_final_state, utterance_final_state = context_state[1], utterance_state[1]

    generated_response = tf.matmul(context_final_state, M)
    generated_response = tf.expand_dims(generated_response, axis=2)
    utterance_final_state = tf.expand_dims(utterance_final_state, axis=2)
    logits = tf.matmul(generated_response, utterance_final_state, transpose_a=True)
    logits = tf.squeeze(logits, axis=2)
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = tf.concat(tf.split(logits, 10, axis=0), axis=1)
        probs = tf.sigmoid(logits, name='probs')        # probabilities of 10 utterances which consist of 1 ground truth response and 9 distracted response
        predictions = {
            'probability': probs
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in grads_and_vars if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def train(estimator):
    train_input_fn = create_input_fn(FLAGS.train_tfrecords_fname, params, tf.estimator.ModeKeys.TRAIN,
                                     shuffle_and_repeat=True)
    estimator.train(train_input_fn)


def evaluate(estimator):
    eval_input_fn = create_input_fn(FLAGS.dev_tfrecords_fname, params, tf.estimator.ModeKeys.EVAL,
                                    shuffle_and_repeat=True)
    estimator.evaluate(eval_input_fn)

def predict(estimator):
    test_input_fn = create_input_fn(FLAGS.test_tfrecords_fname, params, tf.estimator.ModeKeys.PREDICT,
                                    shuffle_and_repeat=False)
    probs = estimator.predict(test_input_fn)
    for p in probs:
        pred_ind = np.argmax(p)

if __name__ == '__main__':
    vocab = build_vocab()
    params = {
        'batch_size': 32,
        'num_epochs': 10,
        'buffer_size': 100,
        'vocab_size': len(vocab) + 2,    # In the vocabulary, index 0 stands for PAD and index 1 stands for OOV
        'model_dir': 'model',
    }
    cfg = tf.estimator.RunConfig(save_checkpoints_steps=10)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params['model_dir'],
        params=params,
        config=cfg
    )
    predict(estimator)





