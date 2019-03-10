import os
import csv
import tensorflow as tf
from collections import defaultdict
import re
import pickle
from config import FLAGS

punctuation_pattern = re.compile(r'[,，。?!]')
PAD = 0
OOV = 1


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


def create_example_test(line, vocab):
    context, utterance = line[:2]
    distractors = line[2:]

    context_inds = transform_sentence(context, vocab, FLAGS.max_context_length)
    utterance_inds = transform_sentence(utterance, vocab, FLAGS.max_utterance_length)
    utterance_inds = utterance_inds + (FLAGS.max_utterance_length - len(utterance_inds)) * [0]
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
        distractor_inds = distractor_inds + (FLAGS.max_utterance_length - len(distractor_inds)) * [0]
        distractor_length = len(distractor_inds)
        example.features.feature[dis_key].int64_list.value.extend(distractor_inds)
        example.features.feature[dis_key_length].int64_list.value.extend([distractor_length])

    return example


def create_tfrecords(input_fname, output_fname, example_fn, vocab):
    if not os.path.exists(output_fname):
        writer = tf.python_io.TFRecordWriter(output_fname)
        with open(input_fname, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            next(reader)
            for lno, line in enumerate(reader):
                example = example_fn(line, vocab)
                writer.write(example.SerializeToString())
                if lno % 10000 == 0:
                    print('successfully processed {0} items'.format(lno))
            writer.close()


if __name__ == '__main__':
    vocab = build_vocab()
    create_tfrecords(FLAGS.train_path, FLAGS.train_tfrecords_fname, create_example_train, vocab)
    create_tfrecords(FLAGS.dev_path, FLAGS.dev_tfrecords_fname, create_example_train, vocab)
    create_tfrecords(FLAGS.test_path, FLAGS.test_tfrecords_fname, create_example_test, vocab)










