#first: store audio data in TBL_AUDIO table
import pymysql.cursors
import time
import pandas as pd
import os
path=r'/Users/ruiqianli/Desktop/UOB internship/Speaker_diarization' #Path to store csv files
file=os.path.join(path, "dental_malaya.csv") 
SIT_result=pd.read_csv(file)
SIT_result=SIT_result.fillna(0)


# Connect to the database
connection= pymysql.connect(host='localhost',
                            user='root',
                            password='password',#Here fill in the password set up by the mysql database administrator
                            database='UOBtests',
                            charset='utf8mb4',
                            cursorclass=pymysql.cursors.DictCursor)
with connection:
    with connection.cursor() as cursor:
        # Create a new record
        sql = "INSERT INTO `TBL_AUDIO` ( `audio_name`,`path_orig`,`path_processed`,`create_date`,`create_time`) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(sql, ('dental_malaya','/Users/ruiqianli/Desktop/UOB internship/Speaker_diarization','/Users/ruiqianli/Desktop/UOB internship/Speaker_diarization',time.strftime("%d/%m/%Y"),time.strftime("%H:%M:%S")))

    # connection is not autocommit by default. So you must commit to save your changes.
    connection.commit()
    audio_id_1=cursor.lastrowid


#second: store result data to TBL_STT_RESULT table
#from dataframe"output" to get the SD, STT and label results

connection= pymysql.connect(host='localhost',
                            user='root',
                            password='password',#Here fill in the password set up by the mysql database administrator
                            database='UOBtests',
                            charset='utf8mb4',
                            cursorclass=pymysql.cursors.DictCursor)
with connection:
    with connection.cursor() as cursor:
        # Create a new record
        sql = "INSERT INTO `TBL_STT_RESULT` (`audio_id`, `slice_id`,`start_time`,`end_time`,`duration`,`text`,`speaker_label`,`save_path`,`create_date`,`create_time`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        for i in range(0,len(SIT_result)-1):
            cursor.execute(sql, (audio_id_1,SIT_result.index[i], SIT_result.starttime[i],SIT_result.endtime[i],SIT_result.duration[i],SIT_result.text[i],SIT_result.label[i],'/Users/ruiqianli/Desktop/UOB internship/Speaker_diarization',time.strftime("%d/%m/%Y"),time.strftime("%H:%M:%S")))

    # connection is not autocommit by default. So you must commit to save your changes.
    connection.commit()
