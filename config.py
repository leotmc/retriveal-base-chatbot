import tensorflow as tf

tf.flags.DEFINE_integer('min_word_freq', 20, 'minimal word frequency in vocabulary')
tf.flags.DEFINE_integer('max_sentence_length', 30, 'maximal length of the sentence')
tf.flags.DEFINE_integer('num_classes', 2, 'number of classes')
tf.flags.DEFINE_integer('hidden_size', 70, 'number of hidden units')
tf.flags.DEFINE_integer('embedding_size', 128, 'embedding size')
tf.flags.DEFINE_integer('max_context_length', 160, 'maximal length of context')
tf.flags.DEFINE_integer('max_utterance_length', 80, 'maximal length of utterance')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.flags.DEFINE_string('train_path', 'data/train.csv', 'training data path')
tf.flags.DEFINE_string('dev_path', 'data/valid.csv', 'development data path')
tf.flags.DEFINE_string('test_path', 'data/test.csv', 'test data path')
tf.flags.DEFINE_string('vocab_path', 'data/vocab.pickle', 'vocabulary path')
tf.flags.DEFINE_string('train_tfrecords_fname', 'data/train.tfrecords', 'tfrecords filename of training data')
tf.flags.DEFINE_string('dev_tfrecords_fname', 'data/dev.tfrecords', 'tfrecords filename of development data')
tf.flags.DEFINE_string('test_tfrecords_fname', 'data/test.tfrecords', 'tfrecords filename of test data')
FLAGS = tf.flags.FLAGS

