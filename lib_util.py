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

import pickle
import tensorflow.keras.backend as K
import json
import os


def load_from_bat(lst_argv):
    try:
        if bool(lst_argv[5]):
            assert len(lst_argv) == 6
            hparams = json.loads(lst_argv[1])
            data_path = lst_argv[2]
            save_dir = lst_argv[3]
            num_samples = lst_argv[4]
            training = lst_argv[5]
            return hparams, data_path, save_dir, int(num_samples), bool(training)
    except:
        if bool(lst_argv[2]):
            assert len(lst_argv) == 3
            translate_sentence = lst_argv[1]
            return translate_sentence



def save_dicts(enc_dic, dec_dic, enc_max_seq_len, dec_max_seq_len):
    with open('C:\\Users\Jesse Berdowski\dicts.pickle', 'wb') as f:
        dikt = dict(enc_dict=enc_dic,
                    dec_dict=dec_dic,
                    enc_max_seq_len=enc_max_seq_len,
                    dec_max_seq_len=dec_max_seq_len)
        pickle.dump(dikt, f)


def save_model_weights(enc_w, dec_w):
    with open('C:\\Users\Jesse Berdowski\weights.pickle', 'wb') as f:
        lst = []
        for item in enc_w:
            lst.append(K.eval(item))
        lst_ = []
        for item in dec_w:
            lst_.append(K.eval(item))
        dikt = dict(enc_weights=lst,
                    dec_weights=lst_)
        pickle.dump(dikt, f)


def load_pickle(path, sort):
    with open(path, 'rb') as f:
        dikt = pickle.load(f)
        if sort is 'dicts':
            return dikt['enc_dict'], dikt['dec_dict'], dikt['enc_max_seq_len'], dikt['dec_max_seq_len'],
        elif sort is 'weights':
            return dikt['enc_weights'], dikt['dec_weights']


def get_dir(data_path):
    return os.path.join(os.path.dirname(__file__), data_path)

