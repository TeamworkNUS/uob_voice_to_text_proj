@echo off
echo model download start...
echo ###################################################################################
wget https://alphacephei.com/kaldi/models/vosk-model-en-us-0.22.zip --no-check-certificate
###################################################################################
echo model download success!
echo ###################################################################################
echo unzip model...
jar xf vosk-model-en-us-0.22.zip
echo ###################################################################################
echo move model to folder...
MOVE vosk-model-en-us-0.22 model
echo success!
pause