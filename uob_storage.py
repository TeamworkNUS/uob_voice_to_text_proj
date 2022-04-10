from datetime import date, datetime
import os
import time
from pymysql import Date
import pymysql.cursors
import uob_utils
from init import (
    dbHost,
    dbName,
    dbUser,
    dbPwd,
    AUDIO_NAME
)

def connectDB():
    connection= pymysql.connect(host=dbHost,
                                user=dbUser,
                                password= dbPwd,#Here fill in the password set up by the mysql database administrator
                                database= dbName,
                                charset='utf8mb4',
                                cursorclass=pymysql.cursors.DictCursor)
    return connection


def dbInsertAudio(upload_filename, upload_filepath, audioname, audiopath):
    connection= connectDB()
    
    with connection:
        with connection.cursor() as cursor:
        #zero: retrieve Versions from TBL_VERSIONS table, in case of Insert conflict
            audio_id_ver=''
            sqlS = "SELECT VERSION_VALUE FROM `uobtests`.TBL_VERSIONS WHERE VERSION_NAME='AUDIO_ID_VER';"
            cursor.execute(sqlS)
            audio_id_ver = cursor.fetchone()['VERSION_VALUE']
            # print('audio_id_ver: ', audio_id_ver, type(audio_id_ver))
            audio_id_ver = int(audio_id_ver)
            
        #first: store result data to TBL_AUDIO table
            # Create a new record
            audio_id = 'AUDIO_%s_%04d'%(time.strftime("%Y%m%d"),audio_id_ver) # 5+1+8+1+4=19;
            audio_meta = uob_utils.get_audio_meta(audioname=audioname, audiopath=audiopath)
            sql = "INSERT INTO `TBL_AUDIO` ( `audio_id`,`audio_name`,`path_orig`,`upload_filename`,`path_upload`,`audio_meta`,`create_by`,`create_date`,`create_time`,`update_by`,`update_date`,`update_time`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
            cursor.execute(sql, (audio_id, audioname, audiopath, upload_filename, upload_filepath, audio_meta, str(os.getlogin()),time.strftime("%Y-%m-%d"),time.strftime("%H:%M:%S"), str(os.getlogin()),time.strftime("%Y-%m-%d"),time.strftime("%H:%M:%S")))
        
        #second: update TBL_VERSIONS
            sqlU = "UPDATE TBL_VERSIONS SET VERSION_VALUE=VERSION_VALUE+1 WHERE VERSION_NAME='AUDIO_ID_VER';"
            cursor.execute(sqlU)
            cursor.execute('COMMIT;')
            
        connection.commit()
    return audio_id
    
    
def dbInsertSTT(finalDf, audio_id, slices_path):
    connection= connectDB()
                        
    # resultDf=finalDf.fillna(0)
    resultDf = finalDf
    with connection:
        with connection.cursor() as cursor:
            # connection is not autocommit by default. So you must commit to save your changes.
            # connection.commit()
            # audio_id_1=cursor.lastrowid
        
        #second: store result data to TBL_STT_RESULT table
        #from dataframe"output" to get the SD, STT and label results
            # Create a new record
            sql = "INSERT INTO `TBL_STT_RESULT` (`audio_id`, `slice_id`,`start_time`,`end_time`,`duration`,`text`,`speaker_label`,`slice_name`,`slice_path`,`create_by`,`create_date`,`create_time`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            for i in range(0,len(resultDf)-1):
                cursor.execute(sql, (audio_id,resultDf['index'][i], resultDf.starttime[i],resultDf.endtime[i],resultDf.duration[i],resultDf.text[i],resultDf.label[i],resultDf.slice_wav_file_name[i],slices_path,str(os.getlogin()),time.strftime("%Y-%m-%d"),time.strftime("%H:%M:%S")))

        # connection is not autocommit by default. So you must commit to save your changes.
        connection.commit()


def dbUpdateAudio_processedInfo(audio_id, **kwargs): #audioname_processed, path_processed,
    connection= connectDB()
    
    with connection:
        with connection.cursor() as cursor:
            sql = "UPDATE TABLE `TBL_AUDIO` SET audio_name_processed = %s , path_processed = %s WHERE audio_id = %s;"
            cursor.execute(sql,(kwargs.get('audioname_processed'), kwargs.get('path_processed'), audio_id))
        connection.commit()
    


def dbInsertLog(audio_id, params, analysis_name, process_time):
    connection= connectDB()
    
    with connection:
        with connection.cursor() as cursor:
            log_id = 'log_'+time.strftime("%Y%m%d_%H%M%S")
            sql = "INSERT INTO `TBL_PROCESS_LOG` (`log_id`,`audio_id`,`params`,`analysis_name`,`process_time`,`create_by`,`create_on`) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (log_id, audio_id, params, analysis_name, process_time, str(os.getlogin()),date.today()))
            
        connection.commit()
    