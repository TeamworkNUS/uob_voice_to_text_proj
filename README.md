# uob_voice_to_text_proj
1.	‘spectralcluster’ package version violation
When spectralcluster==0.2.4 and use Malaya-speech Diarization models, system will give error like below.


Problem:
!pip install spectralcluster==0.1.0   #-->malaya-speech
!pip install spectralcluster==0.2.4   #-->pyannote.audio

Solution: 
manually update the ‘diarization.py’ file in Malaya-speech package (locally). Adjust the codes shown in below screenshot.
 

Workaround: 
Manually upgrade/downgrade spectralcluster package version accordingly.
Meanwhile, comment/uncomment below highlighted line accordingly. (Comment when using Malaya-speech [spectralcluster==0.1.0]; Uncomment when using Pyannote.audio [spectralcluster==0.2.4])



2.	SD models in Malaya-speech
‘malaya-speech’ package has many speaker diarization pretrained models. You can setup which model to be used in ‘uob_mainprocess.py’ by simply comment/uncomment the codes.
 
 
 3. For error with package "pyemd" may meet, please read doc "pyemd_problem_solution.docx".
 
 4. Need to download "punkt" from nltk package, can enter:
 
 python -c "import nltk; nltk.download('punkt')"
 
 in cmd
 
 OR
 
 unzipped "punkt.zip" in path C:\Users\《youraccount》\AppData\Roaming\nltk_data\tokenizers
 
 may need create new folder 'tokenizers', make sure the final path of punkt folder is:
 
 C:\Users\《youraccount》\AppData\Roaming\nltk_data\tokenizers\punkt
 
 E.g. C:\Users\hp\AppData\Roaming\nltk_data\tokenizers\punkt



