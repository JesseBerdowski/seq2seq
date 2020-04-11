@echo off

:: Hyperparameters
SET /A embed_dim=256
SET /A batch_size=64
SET /A epochs=100
SET /A units=32
SET optimizer=adam
SET  learning_rate=1e-3
SET loss=categorical_crossentropy
SET shuffle=True
::data args, paths are relative to the s2s_train.py filepath
SET data_path=static\fra.txt
SET save_dir=static
::training
SET /A num_samples=128
SET training=True

:: DO NOT EDIT
set hyperparameters="{\"embed_dim\": %embed_dim%, \"batch_size\": %batch_size%, \"epochs\": %epochs%, \"units\": %units%, \"learning_rate\": %learning_rate%, \"optimizer\": \"%optimizer%\", \"loss\": \"%loss%\", \"shuffle\": \"%shuffle%\"}"
python s2s_train.py %1 %hyperparameters% %data_path% %save_dir% %num_samples% %training%