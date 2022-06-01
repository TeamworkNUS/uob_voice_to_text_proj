from datetime import date
import os
import time
from django.db import models

from uob_website.settings import MEDIA_ROOT


class Audio(models.Model):
    class Meta:
        managed=False
    audio_id = models.CharField(primary_key=True,max_length=30)
    audio_name = models.CharField(max_length=100,null=False,default='')
    path_orig = models.CharField(max_length=150,null=False,default='')
    # audio_file = models.FileField(blank=True, null=True, upload_to=MEDIA_ROOT)
    audio_name_processed = models.CharField(max_length=100,null=True,blank=True)
    path_processed = models.CharField(max_length=150,null=True,blank=True)
    upload_filename = models.CharField(max_length=100,null=False,default='')
    path_upload = models.CharField(max_length=150,null=False,default='')
    upload_file_count = models.IntegerField(default=1)
    description = models.TextField(blank=True,null=True) #models.CharField(max_length=200,null=True)
    audio_meta = models.CharField(max_length=200,blank=True,null=True)
    analysis = models.CharField(max_length=200,blank=True,null=True)
    flg_delete = models.CharField(max_length=1,blank=True,null=True)
    create_by = models.CharField(max_length=50,null=False,default=str(os.getlogin()))
    create_date = models.DateField(null=False,default=time.strftime("%Y-%m-%d"))
    create_time = models.TimeField(null=False,default=time.strftime("%H:%M:%S"))
    update_by = models.CharField(max_length=50,null=False,default=str(os.getlogin()))
    update_date = models.DateField(null=False,default=time.strftime("%Y-%m-%d"))
    update_time = models.TimeField(null=False,default=time.strftime("%H:%M:%S"))


    def __str__(self):
        return self.audio_name
    # def __unicode__ (self):
    #     return  u' %s '%(self.audio_name)
    


class STTresult(models.Model):
    class Meta:
        unique_together=('audio_id','slice_id')
        managed = False
    
    audio_slice_id = models.CharField(primary_key=True,max_length=30)
    audio_id = models.CharField(max_length=30) #models.ForeignKey(Audio, on_delete=models.CASCADE)# 
    slice_id = models.IntegerField() 
    start_time = models.FloatField(null=False,default=0)
    end_time = models.FloatField(null=False,default=0)
    duration = models.FloatField(null=False,default=0)
    speaker_label = models.CharField(max_length=20,null=False,default='') 
    text = models.CharField(max_length=5000,null=True,blank=True)
    slice_name = models.CharField(max_length=100,null=False,default="")
    slice_path = models.CharField(max_length=150,null=False,default="")
    create_by = models.CharField(max_length=50,null=False,default=str(os.getlogin()))
    create_date = models.DateField(null=False,default=time.strftime("%Y-%m-%d"))
    create_time = models.TimeField(null=False,default=time.strftime("%H:%M:%S"))
    
    def __str__(self):
        return self.slice_name
    
    def slice_path_last_folder(self):
        folder_tree_list =  self.slice_path.split('/')
        last_folder = folder_tree_list[-1]
        return last_folder


    

class AnalysisSelection(models.Model):
    class Meta:
        managed=False
    analysisSelection_id = models.AutoField(primary_key=True)
    # analysisSelection_id = models.IntegerField(primary_key=True)
    analysis_name = models.CharField(max_length=50,null=False, default="None", unique=True)
    
    def __str__(self):
        return self.analysis_name


# class ProcessLog(models.model):
#     class Meta:
#         managed=False
#     log_id = models.CharField(primary_key=True, max_length=30,null=False)
#     audio_id = models.CharField(max_length=30)
#     params = models.CharField(max_length=200)
#     analysis_name = models.CharField(max_length=200)
#     message = models.CharField(max_length=200)
#     process_time = models.CharField(max_length=200)
#     create_by = models.CharField(max_length=50,null=False,default="undetect user")
#     create_date = models.DateField(null=False,default=time.time())
    

class Upload(models.Model):
    upload_id = models.CharField(primary_key=True,max_length=30)
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(blank=True, null=False, upload_to=MEDIA_ROOT)
    uploaded_by = models.CharField(max_length=50,null=False,default=str(os.getlogin()))
    uploaded_at = models.DateTimeField(null=False, auto_now_add=True) #default=date.today()
    
    def __str__(self):
        return self.description


class Version(models.Model):
    class Meta:
        managed=False
    version_id = models.AutoField(primary_key=True)
    version_name = models.CharField(max_length=30, unique=True)
    version_value = models.CharField(max_length=30)
    
    def __str__(self):
        return self.version_name+": "+self.version_value
    
        