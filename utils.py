import os
import csv
import tensorflow as tf
from collections import defaultdict
import re
import pickle
import numpy as np
import logging
import sys

tf.flags.DEFINE_integer('min_word_freq', 20, 'minimal word frequency in vocabulary')
tf.flags.DEFINE_integer('max_sentence_length', 30, 'maximal length of the sentence')
tf.flags.DEFINE_integer('num_classes', 2, 'number of classes')
tf.flags.DEFINE_integer('hidden_size', 70, 'number of hidden units')
tf.flags.DEFINE_integer('embedding_size', 128, 'embedding size')
tf.flags.DEFINE_integer('max_context_length', 60, 'maximal length of context')
tf.flags.DEFINE_integer('max_utterance_length', 80, 'maximal length of utterance')
tf.flags.DEFINE_string('train_path', 'data/train.csv', 'training data path')
tf.flags.DEFINE_string('dev_path', 'data/valid.csv', 'development data path')
tf.flags.DEFINE_string('test_path', 'data/test.csv', 'test data path')
tf.flags.DEFINE_string('vocab_path', 'data/vocab.pickle', 'vocabulary path')
tf.flags.DEFINE_string('train_tfrecords_fname', 'data/train.tfrecords', 'tfrecords filename of training data')
tf.flags.DEFINE_string('dev_tfrecords_fname', 'data/dev.tfrecords', 'tfrecords filename of development data')
tf.flags.DEFINE_string('test_tfrecords_fname', 'data/test.tfrecords', 'tfrecords filename of test data')
FLAGS = tf.flags.FLAGS


punctuation_pattern = re.compile(r'[,，。?!]')
PAD = 0
OOV = 1

# Logging
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def format_text(text):
    text = punctuation_pattern.sub('', text)
    return text


def build_vocab():
    if not os.path.exists(FLAGS.vocab_path):
        with open(FLAGS.train_path, 'r') as f:
            vocab = defaultdict(int)
            for lno, line in enumerate(csv.reader(f)):
                if lno == 0:
                    continue
                context, utterance = line[0], line[1]
                words = context.split() + utterance.split()
                for word in words:
                    vocab[word] += 1
                if lno % 10000 == 0:
                    print('{0} sentences processed'.format(lno))
            vocab = [word for word, freq in vocab.items() if freq > FLAGS.min_word_freq]
            vocab = {word: idx + 2 for idx, word in enumerate(vocab)}
            with open(FLAGS.vocab_path, 'wb') as f:
                pickle.dump(vocab, f)
            return vocab
    else:
        with open(FLAGS.vocab_path, 'rb') as f:
            return pickle.load(f)


def transform_sentence(text, vocab, max_length):
    return [vocab.get(word, 1) for i, word in enumerate(text.split()) if i < max_length]


def create_example_train(line, vocab):
    context, utterance, label = line
    context_inds, utterance_inds = transform_sentence(context, vocab, FLAGS.max_context_length), transform_sentence(utterance, vocab, FLAGS.max_utterance_length)
    label = [1] if label == '0' else [0]
    example = tf.train.Example()
    example.features.feature['context'].int64_list.value.extend(context_inds)
    example.features.feature['utterance'].int64_list.value.extend(utterance_inds)
    example.features.feature['label'].float_list.value.extend(label)
    example.features.feature['context_length'].int64_list.value.extend([len(context_inds)])
    example.features.feature['utterance_length'].int64_list.value.extend([len(utterance_inds)])
    return example


def create_example_eval(line, vocab):
    context, utterance = line[:2]
    distractors = line[2:]

    context_inds = transform_sentence(context, vocab, FLAGS.max_context_length)
    utterance_inds = transform_sentence(utterance, vocab, FLAGS.max_utterance_length)
    context_length = len(context_inds)
    utterance_length = len(utterance_inds)

    example = tf.train.Example()
    example.features.feature['context'].int64_list.value.extend(context_inds)
    example.features.feature['utterance'].int64_list.value.extend(utterance_inds)
    example.features.feature['context_length'].int64_list.value.extend([context_length])
    example.features.feature['utterance_length'].int64_list.value.extend([utterance_length])

    for i, distractor in enumerate(distractors):
        dis_key = 'dis_{}'.format(i)
        dis_key_length = 'dis_{}_length'.format(i)
        distractor_inds = transform_sentence(distractor, vocab, FLAGS.max_utterance_length)
        distractor_length = len(distractor_inds)
        example.features.feature[dis_key].int64_list.value.extend(distractor_inds)
        example.features.feature[dis_key_length].int64_list.value.extend([distractor_length])

    return example


def create_tfrecords(input_fname, output_fname, example_fn, vocab):
    if not os.path.exists(output_fname):
        writer = tf.python_io.TFRecordWriter(output_fname)
        with open(input_fname) as f:
            reader = csv.reader(f)
            next(reader)
            for lno, line in enumerate(reader):
                example = example_fn(line, vocab)
                writer.write(example.SerializeToString())
                if lno % 10000 == 0:
                    print('successfully processed {0} items'.format(lno))
            writer.close()


def parse_function(example_proto):
    features = {
        'context': tf.VarLenFeature(dtype=tf.int64),
        'utterance': tf.VarLenFeature(dtype=tf.int64),
        'context_length': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        'utterance_length': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        'label': tf.VarLenFeature(dtype=tf.float32)
    }
    example = tf.parse_single_example(example_proto, features)
    example['context'] = tf.sparse.to_dense(example['context'])
    example['utterance'] = tf.sparse.to_dense(example['utterance'])
    example['label'] = tf.sparse.to_dense(example['label'])
    # tf.Example only supports tf.int64, so we need to cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t
    return example


def create_input_fn(fname, params, shuffle_and_repeat=False):
    def input_fn():
        dataset = tf.data.TFRecordDataset([fname])
        dataset = dataset.map(parse_function)
        dataset = dataset.padded_batch(params.get('batch_size', 1), padded_shapes={'context': [None], 'utterance': [None], 'context_length': (), 'utterance_length': (), 'label': [None]})
        if shuffle_and_repeat:
            dataset = dataset.shuffle(buffer_size=params['buffer_size']).repeat(params['num_epochs'])
        return dataset

    return input_fn


def model_fn(features, labels, mode, params):
    context, utterance, context_length, utterance_length, labels = features['context'], features['utterance'], features['context_length'], features['utterance_length'], features['label']
    embedding_matrix = tf.Variable(tf.random_normal(shape=[params['vocab_size'], params['embedding_size']], stddev=0.01), name='embedding_matrix')
    M = tf.Variable(tf.random_normal(shape=[params['hidden_size'], params['hidden_size']]), name='M')
    context_embedded = tf.nn.embedding_lookup(embedding_matrix, context, name='context_embedded')
    utterance_embedded = tf.nn.embedding_lookup(embedding_matrix, utterance, name='utterance_embedded')

    context_embedded = tf.transpose(context_embedded, perm=[1, 0, 2]) # time major in LSTMBlockFusedCell
    utterance_embedded = tf.transpose(utterance_embedded, perm=[1, 0, 2])

    with tf.variable_scope('context') as vc:
        context_lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_size'], name='context_lstm_cell')
        _, context_state = context_lstm_cell(context_embedded, sequence_length=context_length, dtype=tf.float32)
        # context_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.hidden_size, initializer=tf.orthogonal_initializer)
        # _, context_state = tf.nn.dynamic_rnn(context_lstm_cell, context_embedded, sequence_length=context_length, dtype=tf.float32)
    with tf.variable_scope('utterance') as vu:
        utterance_lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_size'], name='utterance_lstm_cell')
        _, utterance_state = utterance_lstm_cell(utterance_embedded, sequence_length=utterance_length, dtype=tf.float32)
        # utterance_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.hidden_size, initializer=tf.orthogonal_initializer)
        # _, utterance_state = tf.nn.dynamic_rnn(utterance_lstm_cell, utterance_embedded, sequence_length=utterance_length, dtype=tf.float32)
    context_final_state, utterance_final_state = context_state[1], utterance_state[1]

    generated_response = tf.matmul(context_final_state, M)
    generated_response = tf.expand_dims(generated_response, axis=2)
    utterance_final_state = tf.expand_dims(utterance_final_state, axis=2)
    logits = tf.matmul(generated_response, utterance_final_state, transpose_a=True)
    logits = tf.squeeze(logits, axis=2)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in grads_and_vars if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


vocab = build_vocab()
params = {
    'batch_size': 32,
    'num_epochs': 10,
    'buffer_size': 100,
    'vocab_size': len(vocab) + 2,
    'embedding_size': 128,
    'hidden_size': 70,
    'model_dir': 'model',
    'learning_rate': 0.001,
}

cfg = tf.estimator.RunConfig(save_checkpoints_steps=10)
input_fn = create_input_fn(FLAGS.train_tfrecords_fname, params, shuffle_and_repeat=True)
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=params['model_dir'],
    params=params,
    config=cfg
)
estimator.train(input_fn)








