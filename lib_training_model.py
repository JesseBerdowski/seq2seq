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
from lib_layers import Encoder, Decoder


class SequenceToSequenceWithAttention(tf.keras.Model):
    def __init__(self, enc_feats, dec_feats, training, **hparams):
        super(SequenceToSequenceWithAttention, self).__init__()
        self.training = training
        self.dec_feats = dec_feats
        self.encoder = Encoder(feats=enc_feats, embed_dim=hparams['embed_dim'], units=hparams['units'])
        self.decoder = Decoder(feats=dec_feats, units=hparams['units'])

    def call(self, inputs):
        enc_out, enc_state = self.encoder(inputs[0])
        return self.decoder(inputs[1], enc_out, enc_state, self.training)

