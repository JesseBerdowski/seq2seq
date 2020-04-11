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

from lib_translator_model import Translator
from lib_util import get_dir
import sys

with open(get_dir('static/translate.txt'), 'r') as t:
    sentences_in = t.readlines()

if __name__ == '__main__':
    translator = Translator()
    _ = translator(sentences_in)
    for i, translation in enumerate(translator.translate()):
        print('we translated: \'{}\' from your requested sentence\'{}\''.format(translation, sentences_in[i]))
