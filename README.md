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
 





