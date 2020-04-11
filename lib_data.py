# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
from lib_util import get_dir


class Dataset:
    def __init__(self, num_samples):
        self.encoder_lines, self.decoder_lines = self._read_data(get_dir('static/training_set.txt'), num_samples)
        self.encoder_dict, self.decoder_dict = self.get_sorted_dict()
        self.enc_num_feats = len(self.encoder_dict)
        self.dec_num_feats = len(self.decoder_dict)
        self.enc_arr = self.build_embedding_array(self.encoder_lines, self.encoder_dict)
        self.dec_hot_arr = self.build_target_hot_array(self.decoder_lines, self.decoder_dict)

    def __call__(self):
        assert len(self.decoder_lines) == len(self.encoder_lines)
        return self.create_dataset_obj(self.enc_arr, self.dec_hot_arr)

    def _read_data(self, data_path, num_samples):
        encoder_lines, decoder_lines = [], []
        with open(data_path, 'r', encoding='utf-8') as t:
            lines = t.readlines()
        split_lines = [line.split('\t') for line in lines[:num_samples]]
        for split_line in split_lines:
            encoder_lines.append(split_line[0])
            decoder_lines.append('<start> {} <stop>'.format(split_line[1]))
        return self._clean_sentences(encoder_lines), self._clean_sentences(decoder_lines)

    @staticmethod
    def _clean_sentences(lines):
        punct_lst = ['.', '?', '!']
        clean_lines = []
        replace = None
        for line in lines:
            replace = False
            for punct in punct_lst:
                if punct in line and not replace:
                    _ = line.replace(punct, ' {} '.format(punct))
                    clean_line = _.replace("  ", " ")
                    clean_lines.append(clean_line)
                    replace = True
            if not replace:
                clean_lines.append(line)
        return clean_lines

    @property
    def _get_all_tokens_sorted(self):
        encoder_tokens, decoder_tokens = set(), set()
        for encoder_line, decoder_line in zip(self.encoder_lines, self.decoder_lines):
            [encoder_tokens.add(word) for word in encoder_line.split()]
            [decoder_tokens.add(word) for word in decoder_line.split()]
        return sorted(list(encoder_tokens)), sorted(list(decoder_tokens))

    @staticmethod
    def _get_max_seq_len(lines):
        return max(len(line.split()) for line in lines)

    def get_sorted_dict(self):
        encoder_tokens, decoder_tokens = self._get_all_tokens_sorted
        enc_sorted_dict = {}
        dec_sorted_dict = {}
        for i, enc_token in enumerate(encoder_tokens):
            enc_sorted_dict[enc_token] = i
        for i, dec_token in enumerate(decoder_tokens):
            dec_sorted_dict[dec_token] = i
        enc_sorted_dict['<null>'] = len(enc_sorted_dict)
        dec_sorted_dict['<null>'] = len(dec_sorted_dict)
        return enc_sorted_dict, dec_sorted_dict

    def build_embedding_array(self, lines, dikt):
        self.enc_max_seq_len = self._get_max_seq_len(lines)
        embedded_array = np.zeros(shape=(len(lines), self.enc_max_seq_len))
        for i, line in enumerate(lines):
            for j, word in enumerate(line.split()):
                embedded_array[i][j] = dikt[word]
            embedded_array[i][j + 1:] = dikt['<null>']
        return embedded_array

    def build_target_hot_array(self, lines, dikt):
        self.dec_max_seq_len = self._get_max_seq_len(lines)
        hot_array = np.zeros(shape=(len(lines), self.dec_max_seq_len, len(dikt)), dtype='float32')
        for i, line in enumerate(lines):
            for t, index in enumerate(line.split()):
                hot_array[i, t, dikt[index]] = 1
            hot_array[i, t + 1:, dikt['<null>']] = 1
        return hot_array

    def create_dataset_obj(self, enc_embed_arr, dec_hot_arr):
        dataset = tf.data.Dataset.from_tensor_slices((enc_embed_arr, dec_hot_arr)).shuffle(len(self.encoder_lines))
        return dataset


def _build_embedding_array(lines, dikt):
    embedded_array = np.zeros(shape=(len(lines), len(dikt)))
    for i, line in enumerate(lines):
        for j, word in enumerate(line.split()):
            embedded_array[i][j] = dikt[word]
        embedded_array[i][j + 1:] = dikt['<null>']
    return embedded_array


def _clean_sentences(lines):
    assert isinstance(lines, list)
    punct_lst = ['.', '?', '!']
    clean_lines = []
    replace = None
    for line in lines:
        replace = False
        for punct in punct_lst:
            if punct in line and not replace:
                _ = line.replace(punct, ' {} '.format(punct))
                clean_line = _.replace("  ", " ")
                clean_lines.append(clean_line)
                replace = True
        if not replace:
            clean_lines.append(line)
    return clean_lines


def create_inference_array(lines, dikt, target):
    assert isinstance(lines, list)
    start_stop_lines = []
    for line in lines:
        if target:
            start_stop_line = '<start> ' + line + ' <stop>'
        else:
            start_stop_line = line
        start_stop_lines.append(start_stop_line)
    clean_lines = _clean_sentences(start_stop_lines)
    return _build_embedding_array(clean_lines, dikt)


def rev_dec_sorted_dict(dikt):
    rev_dec_sorted_dict = dict()
    for (key, value) in dikt.items():
        rev_dec_sorted_dict[value] = key
    return rev_dec_sorted_dict