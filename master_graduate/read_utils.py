import numpy as np
import copy
import pickle



def batch_generator(arr, n_seqs, n_steps,iterate_times=1):
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    for i in range(iterate_times):
        print(str(i+1)+' epoch')
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


class TextConverter(object):
    def __init__(self, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            file = open('./type/dictionary.txt', 'r')
            vocab = list()
            for line in file.readlines():
                line = line.strip().split(':')
                vocab.append(line[0])
            file.close()
            self.vocab = vocab
            # print(len(self.vocab))
        self.token_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_token_table = dict(enumerate(self.vocab))


    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.token_to_int_table:
            return self.token_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_token_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        # for word in text:
        #     arr.append(self.word_to_int(word))
        codeLine = text.split('\n')
        for line in codeLine:
            line = line.split()
            code = [self.word_to_int(c) for c in line]
            arr.extend(code)
        # print(arr[:10])
        return np.array(arr)

    def arr_to_text(self, index):
        # words = []
        # for index in arr:
        #     words.append(self.int_to_word(index))
        # return "".join(words)
        return self.int_to_word(index)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)
