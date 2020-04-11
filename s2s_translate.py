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
from lib_util import load_from_bat
import sys

sentence_in = load_from_bat(sys.argv)
print(sentence_in)
print(type(sentence_in))

if __name__ == '__main__':
    lst_in = [sentence_in]
    translator = Translator()
    _ = translator(lst_in)
    for i, translation in enumerate(translator.translate()):
        print('we translated: \'{}\' from your requested sentence\'{}\''.format(translation, lst_in[i]))
