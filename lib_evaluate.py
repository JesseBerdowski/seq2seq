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


def set_opt(**hparams):
    opt = hparams['optimizer']
    lr = hparams['learning_rate']
    if not isinstance(lr, float):
        "the learning_rate parameter should written as a negative exponent"
        assert isinstance(lr, float)
    if opt == 'sgd':
        return tf.keras.optimizers.SGD(lr, nesterov=True)
    elif opt == 'adam':
        return tf.keras.optimizers.Adam(lr)
    elif opt == 'adamax':
        return tf.keras.optimizers.Adamax(lr)
    else:
        return "The optimizer isn't recognized as such, try changing its name to" \
               " \'sgd\', \'adam\', or \'adamax\'"


def set_loss(loss):

    if loss == 'categorical_crossentropy':
        return tf.keras.losses.CategoricalCrossentropy()
    elif loss == 'kullback_leibler_divergence':
        return tf.keras.losses.KLDivergence()
    else:
        return "The loss function isn't recognized as such, try changing its name to " \
               "\'categorical_crossentropy\' or \'kullback_leibler_divergence\' "


def prediction(pred, rev_dec_dict):
    pred_index = tf.argmax(pred, axis=1)
    return rev_dec_dict[int(pred_index)]





