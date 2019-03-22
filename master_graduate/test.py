import tensorflow as tf
from read_utils import TextConverter, batch_generator
# from bi_lstm_model import CharRNN
from lstm_model import CharRNN
import os
# from IPython import embed
import codecs

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 256, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_seqs', 128, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 50, 'length of one seq')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_integer('embedding_size', 1024, 'size of embedding')

# tf.flags.DEFINE_string('converter_path', './bi_lstm_model/converter.pkl', 'converter.pkl')
# tf.flags.DEFINE_string('checkpoint_path', './bi_lstm_model/', 'checkpoint path')

tf.flags.DEFINE_string('converter_path', './type_lstm_model/converter.pkl', 'converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', './type_lstm_model/', 'checkpoint path')
tf.flags.DEFINE_string('test_file', './type/test_type.txt', 'use this string to start generating')

tf.flags.DEFINE_float('train_keep_prob', 1.0, 'dropout rate during training')


def main(_):
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size,num_seqs=FLAGS.num_seqs,
                    lstm_size=FLAGS.lstm_size,train_keep_prob=FLAGS.train_keep_prob, num_layers=FLAGS.num_layers,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)

    with codecs.open(FLAGS.test_file,encoding='utf-8') as f:
        text = f.read()
    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
    test_accuracy,test_loss = model.test(g)
    print('the test_accuracy : ',test_accuracy)
    print('the test_loss : ',test_loss)


if __name__ == '__main__':
    tf.app.run()
