# seq2seq
An Encoder-Decoder Model with Luong-style Attention

The pretrained model allows the user to translate sentences from English to French.
It was trained on the en-fra dataset from http://www.manythings.org/anki/. 
Any dataset from this page could be used for training.

# How to Use
I wrote premade scripts for training the model and translating with the model.
### Translating with the model
-Create sentences in the static/translate.txt file following the template.
-run the s2s_sample.bat file
### Training the model
-save one of the files of the website as static/training_set.txt
-update the hparams in the s2s_train.bat file
-run the s2s_train.bat file
