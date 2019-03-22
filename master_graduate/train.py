import tensorflow as tf
from read_utils import TextConverter, batch_generator
# from bi_lstm_model import CharRNN
from lstm_model import CharRNN
import os
import codecs

FLAGS = tf.flags.FLAGS

# tf.flags.DEFINE_string('model_path', 'bi_lstm_model', 'name of the model')

tf.flags.DEFINE_string('model_path', 'type_lstm_model', 'name of the model')

tf.flags.DEFINE_integer('num_seqs', 128, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 50, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 256, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_integer('embedding_size', 1024, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', './type/train_type.txt', 'utf8 encoded text file')
tf.flags.DEFINE_integer('iter_times', 2, 'train_data iteration times to train')


def main(_):

    model_path =  FLAGS.model_path
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter()
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps,FLAGS.iter_times)
    # print(converter.vocab_size)
    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    embedding_size=FLAGS.embedding_size
                    )

    model.train(g, model_path)


if __name__ == '__main__':
    tf.app.run()
