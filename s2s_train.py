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

from lib_data import Dataset
from lib_evaluate import set_loss, set_opt
from lib_training_model import SequenceToSequenceWithAttention
from lib_util import save_dicts, save_model_weights, load_from_bat
import sys

hparams, data_path, save_dir, num_samples, training = load_from_bat(sys.argv)

d = Dataset(data_path, num_samples, **hparams)
print(d.dec_hot_arr.shape)
print(d.enc_arr.shape)
dataset = d()
s2s = SequenceToSequenceWithAttention(len(d.encoder_dict), len(d.decoder_dict), training, **hparams)

loss_obj = set_loss(hparams['loss'])
optimizer = set_opt(**hparams)

save_dicts(d.encoder_dict, d.decoder_dict, d.enc_max_seq_len, d.dec_max_seq_len)


def model_fit(**hparams):
    s2s.compile(optimizer=optimizer, loss=loss_obj, metrics=['accuracy'])
    s2s.fit(x=[d.enc_arr, d.dec_hot_arr], y=d.dec_hot_arr, epochs=hparams['epochs'],
            shuffle=hparams['shuffle'], batch_size=hparams['batch_size'])
    save_model_weights(s2s.encoder.trainable_weights, s2s.decoder.trainable_weights)


if __name__ == '__main__':
    model_fit(**hparams)
