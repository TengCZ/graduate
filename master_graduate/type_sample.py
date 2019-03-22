import tensorflow as tf
from read_utils import TextConverter
from pycparser import c_lexer
# from bi_lstm_model import CharRNN

from lstm_model import CharRNN

import os
import numpy as np
from IPython import embed
import codecs

import data_process as dp

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 256, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_integer('embedding_size', 1024, 'size of embedding')

# tf.flags.DEFINE_string('converter_path', './bi_lstm_model/converter.pkl', 'model/converter.pkl')
# tf.flags.DEFINE_string('checkpoint_path', './bi_lstm_model/', 'checkpoint path')

tf.flags.DEFINE_string('converter_path', './type_lstm_model/converter.pkl', 'converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', './type_lstm_model/', 'checkpoint path')

tf.flags.DEFINE_integer('max_length', 1, 'max length to generate')


def main(_):
    # FLAGS.start_string = FLAGS.start_string.decode('utf-8')
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)
    filepath = './data/sample.txt'
    file = open(filepath,'r')
    code = file.readlines()
    source = dp.outOfComment(code)
    print(source)

    token_value = dp.get_token_type(source)
    value_type = dp.get_value_type(source)

    text = " ".join(token_value)
    start = converter.text_to_arr(text)

    tokens = model.sample(start, converter.vocab_size)
    # print(source)
    print('The Top_2')
    type_to_value = dp.get_map('./type/type_to_value.txt')
    for token in tokens:
        # print(converter.arr_to_text(token[0]),' : ',token[1])#token : probility
        res = converter.arr_to_text(token[0])
        if res in type_to_value:
            print(res,' ',type_to_value[res],' ',token[1])
        else:
           ret=[]
           for value in value_type:
               if value_type[value]== res:
                    ret.append(value)
           if len(ret)>0:
               print(res,' ',ret,' ',token[1])
           else:
                print('no result')








if __name__ == '__main__':
    tf.app.run()
