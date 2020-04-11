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

import tensorflow as tf
import numpy as np
from lib_data import create_inference_array, rev_dec_sorted_dict
from lib_layers import Encoder, Decoder
from lib_util import load_pickle
from lib_evaluate import prediction

_defaults = dict(
    latent_dim=256,
    embedding_dim=256,
    batch_size=1,
    epochs=1,
    max_seq_len=None,
    optimizer='adam',
    learning_rate=0.001,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    data_path='C:\\Users\\Jesse Berdowski\\Downloads\\fra.txt',
    save_dir='static\\',
    num_samples=1,
    training=False
)


class Translator:
    def __init__(self):
        enc_weights, self.dec_weights = load_pickle('C:\\Users\\Jesse Berdowski\\weights.pickle', 'weights')
        self.enc_dict, self.dec_dict, \
        self.enc_max_seq_len, self.dec_max_seq_len = load_pickle('C:\\Users\\Jesse Berdowski\\dicts.pickle', 'dicts')
        self.rev_dec_dict = rev_dec_sorted_dict(self.dec_dict)
        self.encoder = Encoder(feats=len(self.enc_dict), embed_dim=256, units=32)
        self.decoder = Decoder(feats=len(self.dec_dict), units=32)
        _, __ = self.encoder(tf.keras.layers.Input(shape=(None,)))
        self.encoder.set_weights(enc_weights)
        self.decoder(tf.keras.layers.Input(shape=(None,)), _, __, training=False)
        self.decoder.set_weights(self.dec_weights)

    def __call__(self, lst_inputs):
        assert isinstance(lst_inputs, list)
        self.lst_inputs = lst_inputs

    def translate(self):
        predicted_sentences = []
        for sentence in self.lst_inputs:
            in_arr = create_inference_array([sentence], self.enc_dict, False)
            enc_out, state_h = self.encoder(in_arr)
            start = np.zeros(shape=(1, self.dec_max_seq_len))
            start[0, self.dec_dict['<start>']] = 1
            predict_sentence = ''
            for _ in range(self.dec_max_seq_len):
                out, state_h = self.decoder(start, enc_out, state_h, False)
                x = prediction(out, self.rev_dec_dict)
                if x is '<stop>' or _ == self.dec_max_seq_len - 1:
                    predicted_sentences.append(predict_sentence.replace('<start>', '').replace('<stop>', ''))
                predict_sentence += x
        return predicted_sentences

