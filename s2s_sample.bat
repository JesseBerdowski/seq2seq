@echo off

:: prompt user for input
set "response="
set /p response=What sentence would you like translated from English to French?
:: DO NOT EDIT
SET training=False

python s2s_translate.py %1 %response% %training%