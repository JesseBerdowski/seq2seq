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


class Encoder(tf.keras.layers.Layer):
    def __init__(self, feats, embed_dim, units):
        super(Encoder, self).__init__()
        self.embedding_enc = tf.keras.layers.Embedding(input_dim=feats, output_dim=embed_dim)
        self.gru_enc = tf.keras.layers.GRU(units=units, return_sequences=True, return_state=True)

    def call(self, inp):
        x = self.embedding_enc(inp)
        out, state_h = self.gru_enc(x)
        return out, state_h

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embedding_enc': self.embedding_enc,
            'gru_enc': self.gru_enc,
        })
        return config


class Decoder(tf.keras.layers.Layer):
    def __init__(self, feats, units):
        super(Decoder, self).__init__()
        self.attention = tf.keras.layers.Attention()
        self.embedding = tf.keras.layers.Embedding(input_dim=feats, output_dim=units, name='emb')
        self.gru = tf.keras.layers.GRU(units=units, return_state=True, name='gru')
        self.dense = tf.keras.layers.Dense(units=feats, activation='softmax', name='dense')

    @tf.function
    def call(self, inputs, enc_out, enc_state, training):
        dec_state = enc_state
        if training:
            for t in range(inputs.shape[1]):
                context_vector = self.attention([dec_state, enc_out])
                context_vector = tf.reduce_sum(context_vector, 1, keepdims=True)
                x = self.embedding(inputs[:, t, :])
                x = x + context_vector
                out, dec_state = self.gru(x, initial_state=dec_state)
                out = self.dense(out)
                if not t:
                    conc_out = tf.expand_dims(out, axis=1)
                else:
                    conc_out = tf.concat((conc_out, tf.expand_dims(out, axis=1)), axis=1)
            return conc_out
        elif not training:
            context_vector = self.attention([dec_state, enc_out])
            context_vector = tf.reduce_sum(context_vector, 1, keepdims=True)
            x = self.embedding(inputs)
            x = x + context_vector
            out, dec_state = self.gru(x, initial_state=dec_state)
            return self.dense(out), dec_state

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'attention': self.attention,
            'embedding': self.embedding,
            'gru': self.gru,
            'dense': self.dense,
        })
        return config
