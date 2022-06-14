import csv
from datetime import datetime
import time
import json
import os
import threading
import queue

from django.http import HttpResponse, Http404, HttpResponseRedirect, JsonResponse, FileResponse
from django.shortcuts import redirect, render, get_object_or_404
from django.template import loader
from django.urls import reverse
from django.views import View
from django.views.generic.edit import DeleteView, CreateView
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import analysis

from uob_website.settings import MEDIA_ROOT

from .forms import AnalysisSelectionForm, UploadModelForm
from .models import AnalysisSelection, Audio, STTresult
from analysis import uob_main, uob_storage, uob_mainprocess, uob_utils
from .uob_init import (
    FLG_REDUCE_NOISE,
    FLG_SPEECH_ENHANCE,
    FLG_SPEECH_ENHANCE_NEW,
    FLG_SUPER_RES,
    sttModel,
    sdModel,
    userguide_path,
)

# Create your views here.
@login_required(login_url="/accounts/login/")
def main(request):
    num_audios_all = Audio.objects.all().count()
    if request.user.is_staff: #is_superuser:
        num_audios= Audio.objects.filter(flg_delete=None).count()
        num_audios_unanalysis = Audio.objects.filter(flg_delete=None, analysis='{}').count() + Audio.objects.filter(flg_delete=None, analysis=None).count()
    else:
        ### Option 1: Using Windows Authentication
        # num_audios= Audio.objects.filter(flg_delete=None, create_by=str(os.getlogin())).count()
        # num_audios_unanalysis = Audio.objects.filter(flg_delete=None, create_by=str(os.getlogin()), analysis='{}').count() + Audio.objects.filter(flg_delete=None, create_by=str(os.getlogin()), analysis=None).count()
        ### Option 2: Using Django Authentication
        num_audios= Audio.objects.filter(flg_delete=None, create_by=str(request.user.username)).count()
        num_audios_unanalysis = Audio.objects.filter(flg_delete=None, create_by=str(request.user.username), analysis='{}').count() + Audio.objects.filter(flg_delete=None, create_by=str(request.user.username), analysis=None).count()

        
    context = {'num_audios': num_audios,
               'num_audios_all': num_audios_all,
               'num_audios_unanalysis': num_audios_unanalysis
               }
    return render(request, "analysis/main.html", context=context)
    

@login_required(login_url="/accounts/login/")
def upload(request):
    # context={}
    # return render(request, template_name='analysis/upload.html', context=context)

    # if request.method == 'POST' and request.FILES['myfile']:
    #     myfile = request.FILES['myfile']
    #     fs = FileSystemStorage()
    #     filename = fs.save(myfile.name, myfile)
    #     uploaded_file_url = fs.url(filename)
    #     return render(request, 'analysis/upload.html', {
    #         'uploaded_file_url': uploaded_file_url
    #     })
    # return render(request, 'analysis/upload.html')
    
    if request.method == 'POST':
        upload_id = uob_storage.dbGenerateUploadId()
        uploadForm = UploadModelForm(request.POST, request.FILES, initial={'upload_id': upload_id})
        if uploadForm.is_valid():
            uploadFile = request.FILES['document']
            # file_type = uploadFile.document.url.split('.')[-1]
            
            fs = FileSystemStorage()
            upload_filename = fs.save(uploadFile.name, uploadFile)
            upload_filepath = MEDIA_ROOT.replace("\\","/")
            upload_file_url = fs.url(upload_filename)
            upload_desc = str(uploadForm.cleaned_data.get("description"))
            
            upload_file_count, upload_filetype_general, upzip_audio_name, upzip_audio_path = uob_mainprocess.process_upload_file(upload_filename=upload_filename, upload_filepath=upload_filepath)
            print("view print: count + filetype_general", upload_file_count, upload_filetype_general)

            print('upload_filename: ',upload_filename)
            print('upload_filepath: ',upload_filepath)
            print('upload_file_url: ',upload_file_url)
            print('description: ',upload_desc)
            
            if upload_filetype_general == 'audio':
                audio = Audio()
                audio.audio_id = uob_storage.dbGenerateAudioId()
                audio.audio_name = upzip_audio_name
                audio.path_orig = upzip_audio_path
                audio.upload_filename = upload_filename
                audio.path_upload = upload_filepath
                audio.upload_file_count = upload_file_count
                audio.audio_meta = uob_utils.get_audio_meta(audioname=audio.audio_name, audiopath=audio.path_orig)
                audio.description = upload_desc
                audio.analysis = {}
                audio.create_date = time.strftime("%Y-%m-%d")
                audio.create_time = time.strftime("%H:%M:%S")
                audio.update_date = time.strftime("%Y-%m-%d")
                audio.update_time = time.strftime("%H:%M:%S")
                audio.create_by = request.user.username
                audio.update_by = request.user.username
                audio.save()
            elif upload_filetype_general == 'compressed':
                for unzip_audioname in os.listdir(upzip_audio_path):
                    audio = Audio()
                    audio.audio_id = uob_storage.dbGenerateAudioId()
                    audio.audio_name = unzip_audioname
                    audio.path_orig = upzip_audio_path
                    audio.upload_filename = upload_filename
                    audio.path_upload = upload_filepath
                    audio.upload_file_count = upload_file_count
                    audio.audio_meta = uob_utils.get_audio_meta(audioname=audio.audio_name, audiopath=audio.path_orig)
                    audio.description = upload_desc
                    audio.analysis = {}
                    audio.create_date = time.strftime("%Y-%m-%d")
                    audio.create_time = time.strftime("%H:%M:%S")
                    audio.update_date = time.strftime("%Y-%m-%d")
                    audio.update_time = time.strftime("%H:%M:%S")
                    audio.create_by = request.user.username
                    audio.update_by = request.user.username
                    audio.save()
                    
                
            # return HttpResponseRedirect(reverse('analysis:upload'))
            uploadForm = UploadModelForm()
            return render(request, 'analysis/upload.html', {'upload_file_url': upload_file_url,'uploadForm':uploadForm})

    else:
        uploadForm = UploadModelForm()
        
    return render(request, 'analysis/upload.html', {'uploadForm': uploadForm})


def about(request):
    context={}
    return render(request, template_name='analysis/about.html', context=context)


@login_required(login_url="/accounts/login/")
def history(request):
    
    ## Get analysis thread and running queue item for current Client user
    t_list = [thread for thread in threading.enumerate() if thread.getName()=='dfAnalysisThread-main-'+request.user.username]
    t = t_list[0] if len(t_list)!=0 else None
    if t != None:
        analysisAudioQueue = t.input_queue
    else:
        analysisAudioQueue = queue.Queue()
    
    if 'queueItem_{}'.format(request.user.username) not in globals():
        print('global var queueItem_{} does not exist'.format(request.user.username))
        globals()['queueItem_{}'.format(request.user.username)] = None
    else:
        print('global var queueItem_{} exists!!!!'.format(request.user.username))
    
    
    ## Start to process POST requests
    if request.method == 'POST':
        ### Single Delete (flag audio.flg_delete as "X")
        if request.POST.get('modal_singleDelete'):
            audio_to_delete = Audio.objects.get(pk=request.POST.get('modal_singleDelete'))
            if audio_to_delete != None:
                audio_to_delete.flg_delete = 'X'
                audio_to_delete.save(update_fields=['flg_delete'])
            
        ### Single Analysis Start
        if request.POST.get('modal_singleStart'):
            audio_to_analysis = Audio.objects.get(pk=request.POST.get('modal_singleStart'))
            if audio_to_analysis != None:
                analysisSelectionForm = AnalysisSelectionForm(request.POST)
                if analysisSelectionForm.is_valid():
                    print('valid checked')
                    analysis_selected = analysisSelectionForm.cleaned_data.get('analysisChoices') #['1', '2']
                    print('analysisChoices: ', analysis_selected)
                    choices = AnalysisSelection.objects.all()
                    json_analysis_selection = {}
                    
                    ### * Start to Analyze
                    threadLock = threading.Lock()
                    threadLock.acquire()
                    try:
                        if t != None:
                            if t.is_alive():
                                t.send(audio_to_analysis)
                            else:
                                analysisAudioQueue.put(audio_to_analysis)
                                t = AnalysisThread(
                                                    analysis_selected=analysis_selected,
                                                    choices=choices,
                                                    json_analysis_selection=json_analysis_selection,
                                                    input_queue=analysisAudioQueue,
                                                    args={'username':request.user.username},
                                                    )
                                t.setName('djAnalysisThread-main-'+request.user.username) # Djanog Authentication # replace with os.getlogin() if Windows Authentication
                                t.start()
                        else:
                            analysisAudioQueue.put(audio_to_analysis)
                            t = AnalysisThread(
                                                analysis_selected=analysis_selected,
                                                choices=choices,
                                                json_analysis_selection=json_analysis_selection,
                                                input_queue=analysisAudioQueue,
                                                args={'username':request.user.username},
                                                )
                            t.setName('djAnalysisThread-main-'+request.user.username) # Djanog Authentication # replace with os.getlogin() if Windows Authentication
                            t.start()
                    finally:
                        threadLock.release()
                        
                else:
                    print('analysis form is not valid')
        
        ### Batch Delete (flag audio.flg_delete as "X")
        if request.POST.get('modal_batchDelete'):
            audioIdList = request.POST.get('modal_batchDelete') #string
            audioIdList = audioIdList.split(",") #list
            audios_to_delete = Audio.objects.filter(pk__in=audioIdList) #queryset
            for audio in audios_to_delete:
                if audio != None:
                    audio.flg_delete = 'X'
                    audio.save(update_fields=['flg_delete'])
        
        ### Batch Analysis Start
        if request.POST.get('modal_batchStart'):
            audioIdList = request.POST.get('modal_batchStart') #string
            audioIdList = audioIdList.split(",") #list
            audios_to_start = Audio.objects.filter(pk__in=audioIdList) #queryset
            
            analysisSelectionForm = AnalysisSelectionForm(request.POST)
            if analysisSelectionForm.is_valid():
                print('analysis form valid checked')
                analysis_selected = analysisSelectionForm.cleaned_data.get('analysisChoices')
                print('analysisChoices: ', analysis_selected)
                choices = AnalysisSelection.objects.all()
                json_analysis_selection = {}
                
                ### * Start to Analyze
                threadLock = threading.Lock()
                threadLock.acquire()
                try:
                    if t != None:
                        if t.is_alive():
                            for audio_to_analysis in audios_to_start:
                                t.send(audio_to_analysis)
                        else:
                            for audio_to_analysis in audios_to_start:
                                analysisAudioQueue.put(audio_to_analysis)
                            t = AnalysisThread(
                                                analysis_selected=analysis_selected,
                                                choices=choices,
                                                json_analysis_selection=json_analysis_selection,
                                                input_queue=analysisAudioQueue,
                                                args={'username':request.user.username},
                                                )
                            t.setName('djAnalysisThread-main-'+request.user.username) # Djanog Authentication # replace with os.getlogin() if Windows Authentication
                            t.start()
                    else:
                        for audio_to_analysis in audios_to_start:
                            analysisAudioQueue.put(audio_to_analysis)
                        t = AnalysisThread(
                                            analysis_selected=analysis_selected,
                                            choices=choices,
                                            json_analysis_selection=json_analysis_selection,
                                            input_queue=analysisAudioQueue,
                                            args={'username':request.user.username},
                                            )
                        t.setName('djAnalysisThread-main-'+request.user.username) # Djanog Authentication # replace with os.getlogin() if Windows Authentication
                        t.start()
                finally:
                    threadLock.release()
                    
            else:
                print('analysis form is not valid')
                    
    else:
        print("not post")   
    
    
    threadCount = threading.activeCount()
    threadList = threading.enumerate()
    threadNameList = [thread.getName() for thread in threadList]
    flag_analysisThread_currUser_active = True if True in [ True for name in threadNameList if 'djAnalysisThread' in name and request.user.username in name] else False
    flag_analysisThread_active = True if True in [ True for name in threadNameList if 'djAnalysisThread' in name ] else False
    activeAnalysis_audioId_list = [audio.audio_id for audio in analysisAudioQueue.queue]
    time.sleep(1) # DO NOT DELETE!!! in case cannot get the running item in globals()
    if globals()['queueItem_{}'.format(request.user.username)] != None: activeAnalysis_audioId_list.append(globals()['queueItem_{}'.format(request.user.username)].audio_id)  # add the running item which was queue.get() from Queue
    ### Option 1: Using Windows Authentication
    # audioList_unanalysis = Audio.objects.filter(flg_delete=None, create_by=str(os.getlogin()), analysis='{}')
    # audioList = Audio.objects.filter(flg_delete=None, create_by=str(os.getlogin()))
    # audioList = audioList.order_by('-create_date','-create_time','-audio_id')
    ### Option 2: Using Django Authentication
    audioList_unanalysis = Audio.objects.filter(flg_delete=None, create_by=str(request.user.username), analysis='{}')
    audioList = Audio.objects.filter(flg_delete=None, create_by=str(request.user.username))
    audioList = audioList.order_by('-create_date','-create_time','-audio_id')
    
    analysisSelectionForm = AnalysisSelectionForm()
    context = {
        'audioList': audioList,
        'analysisSelectionForm': analysisSelectionForm,
        'threadCount': threadCount,
        'threadList': threadList,
        'threadNameList': threadNameList,
        'flag_analysisThread_currUser_active': flag_analysisThread_currUser_active,
        'flag_analysisThread_active': flag_analysisThread_active,
        'activeAnalysis_audioId_list': activeAnalysis_audioId_list,
        'audioList_unanalysis': audioList_unanalysis
    }
    print('flag_analysisThread_active:',flag_analysisThread_active)
    if flag_analysisThread_currUser_active == False: globals()['queueItem_{}'.format(request.user.username)] = None
    return render(request, template_name='analysis/history.html', context=context)



@login_required(login_url="/accounts/login/")
def report(request, audio_id):
    audio = get_object_or_404(Audio,pk=audio_id)
    audio_meta = json.loads(audio.audio_meta)
    zip_folder = os.path.splitext(audio.upload_filename)[0]
    sttResult = STTresult.objects.filter(audio_id = audio_id)
    sttResult = sttResult.order_by("slice_id")
    
    context = {'audio': audio, 
               'audio_meta': audio_meta, 
               'zip_folder': zip_folder,
               'sttResult': sttResult
               }
    
    return render(request, 'analysis/report.html', context=context)



def analysis_selection(request, audio_id):
# class AnalysisSelectionView(CreateView):
    
    def get_cancel_url(self):
        url = reverse('analysis:analysis_selection')
        return url
    
    # def analysis_selection(request, audio_id):
    audio = get_object_or_404(Audio,pk=audio_id)
    
    if request.method == 'POST':
        analysisSelectionForm = AnalysisSelectionForm(request.POST) #, initial={'analysisChoices':[2,]}
        print('POST checked')
        if analysisSelectionForm.is_valid():
            # temp = analysisSelectionForm.fields['analysisChoices'].initial
            temp = analysisSelectionForm.cleaned_data.get('analysisChoices')
            print('temp: ', temp)
            print('valid checked')
            # temp = analysisSelectionForm.fields['analysisChoices'].choices
            # temp:  [(1, 'SD+STT'), (2, 'use case 1'), (3, 'use case 3')]

            analysisChoices = request.POST.getlist('cb_inputs')
            print('analysisChoices: ', analysisChoices)

            # TODO: Start to run analysis processes for audio_id = xxx...
            
            
            context = {
                'audio': audio,
                'form': AnalysisSelectionForm(),
            }
            
            # return HttpResponseRedirect(reverse('analysis:history'))  #'/index/?page=2'
            return render(request, 'analysis/analysis_selection.html', context=context)
        else:
            print('not valid')
    
    else:
        analysisSelectionForm = AnalysisSelectionForm()
        # analysisChoices = request.GET.getlist('analysisChoices')
        analysisChoices = analysisSelectionForm.fields['analysisChoices']
        print('analysisChoices: ', analysisChoices)
        print('analysisChoices.widget_attrs: ', analysisChoices.widget_attrs)
        print('analysisChoices.widget: ', analysisChoices.widget)
        print('analysisChoices.widget.attrs: ', analysisChoices.widget.attrs)
        print('analysisChoices.choices: ', analysisChoices.choices)
        print('analysisChoices.widget.attrs: ', analysisChoices.widget.attrs)
        
    context = {
        'audio': audio,
        'form': analysisSelectionForm,
    }
    return render(request, 'analysis/analysis_selection.html',  context)




class MyThread(threading.Thread):
    def __init__(self,target=None,args=()):
        super(MyThread,self).__init__()
        self.target = target
        self.args = args
    def run(self):
        for i in range(5):
            print("i=",i)
            time.sleep(5)


class AnalysisThread(threading.Thread):
    def __init__(self,analysis_selected,choices,json_analysis_selection,input_queue:queue.Queue, target=None,args=()):
        super(AnalysisThread,self).__init__()
        self.target = target
        self.args = args
        self.analysis_selected = analysis_selected
        self.choices = choices
        self.json_analysis_selection = json_analysis_selection
        self.input_queue = input_queue
    
    def send(self, audio):
        self.input_queue.put(audio)

    def run(self):
        
        while True:
            if self.input_queue.empty():
                break
            
            queueItem = self.input_queue.get()
            globals()['queueItem_{}'.format(self.args['username'])] = queueItem
            print(self.args['username'], 'Running Item: ', queueItem, '; Queue: ', self.input_queue.queue)
            

            starttime = datetime.now()
            sdstt = STTresult.objects.filter(audio_id = queueItem.audio_id)
            print(self.analysis_selected)
            print(self.choices)
            for i in self.analysis_selected:
                i_int = int(i)-1
                self.json_analysis_selection[str(i_int)] = self.choices[i_int].analysis_name  #example: {"0":"SD+STT", "1":"use case 1"}
                if self.choices[i_int].analysis_name == 'SD+STT':
                    print('*'*30)
                    print("SD+STT starts")
                    if not sdstt:
                        try:
                            print("SD+STT running", queueItem.audio_id)
                            uob_main.sd_and_stt(queueItem, starttime, self.choices[i_int].analysis_name, self.args['username'])
                        except Exception as e1:
                            print('SD+STT fails out of Exception:', type(e1))
                            print(e1) 
                    else:
                        # raise ValueError("Audio "+audio_to_analysis.audio_id+" has been performed analysis of "+choices[i_int].analysis_name)
                        print("Warning: "+"Audio "+queueItem.audio_id+" has been performed analysis of "+self.choices[i_int].analysis_name)
                    
                    print("SD+STT ends")
                    print('*'*30)
                    
                if self.choices[i_int].analysis_name == 'use case 1':
                    print('*'*30)
                    print("use case 1 starts")
                    print("use case 1 ends")
                    print('*'*30)
                    
                if self.choices[i_int].analysis_name == 'use case 3':
                    print('*'*30)
                    print("use case 3 starts")
                    print("use case 3 ends")
                    print('*'*30)
                                        
            
            endtime = datetime.now()

            print('*' * 30,'\n  Finished!!',)
            print('start from:', starttime) 
            print('end at:', endtime) 
            print('duration: ', endtime-starttime)
            
            ### * Save to Log Table
            params = json.dumps({"NR":FLG_REDUCE_NOISE, "SE":FLG_SPEECH_ENHANCE, "SE_NEW":FLG_SPEECH_ENHANCE_NEW, "SR":FLG_SUPER_RES, "SD":sdModel, "STT":sttModel})
            analysis_name = json.dumps(self.json_analysis_selection)
            message = ''
            process_time = '{"starttime":"%s", "endtime":"%s", "duration":"%s"}'%(starttime,endtime,endtime-starttime)
            try:
                uob_storage.dbInsertLog(audio_id=queueItem.audio_id, params=params, analysis_name=analysis_name, process_time=process_time, username=self.args['username'])
            except Exception as e_log:
                print('Save log fails out of Exception:', type(e_log))
                print(e_log) 
            
            
            ### * Declare the running task in Queue is done
            self.input_queue.task_done()
            
        return


def stt_exportcsv(request, audio_id):
    sttResult = STTresult.objects.filter(audio_id = audio_id)
    sttResult = sttResult.order_by("slice_id")
    opts = sttResult.model._meta
    filename = 'STT_Result_'+audio_id+'.csv'
    response = HttpResponse('text/csv')
    response['Content-Disposition'] = 'attachment; filename=%s'%(filename)
    writer = csv.writer(response)
    
    ### Option1: Write all cols in database table
    # ## Write data header
    # field_names = [field.name for field in opts.fields]
    # writer.writerow(field_names)
    # ## Write data rows
    # for obj in sttResult:
    #     writer.writerow([getattr(obj, field) for field in field_names])
    
    ### Option2: Write partial cols in database table
    ## Write data header
    field_names = ['audio_id','slice_id','speaker_label','start_time','end_time','duration','text']
    writer.writerow(field_names)
    ## Write data rows
    sttResult_tbl = sttResult.values_list('audio_id','slice_id','speaker_label','start_time','end_time','duration','text')
    for slice in sttResult_tbl:
        writer.writerow(slice)
        
    return response

            
def download_userguide(request):
    file_server = userguide_path
    if not os.path.exists(file_server):
        messages.error(request, 'UserGuide is not fount. Please contact the administrator.')
    else:
        file_to_download = open(str(file_server), 'rb')
        file_export_name = 'Voice_to_Text_Analysis_Web_UserGuide.pptx'
        response = FileResponse(file_to_download, content_type='application/force-download')
        response['Content-Disposition'] = 'inline; filename=%s'%(file_export_name)
        return response
    return redirect(reverse('analysis:about'))
    