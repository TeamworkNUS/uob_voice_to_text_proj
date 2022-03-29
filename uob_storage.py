from datetime import date, datetime
import os
import time
from pymysql import Date
import pymysql.cursors
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


def dbInsertSTT(finalDf, orig_path, processed_path, slices_path):
    connection= connectDB()
                        
    # resultDf=finalDf.fillna(0)
    resultDf = finalDf
    with connection:
        with connection.cursor() as cursor:
        #first: store result data to TBL_AUDIO table
            # Create a new record
            sql1 = "INSERT INTO `TBL_AUDIO` ( `audio_name`,`path_orig`,`path_processed`,`create_by`,`create_date`,`create_time`) VALUES (%s, %s, %s, %s, %s, %s)"
            cursor.execute(sql1, (os.path.splitext(AUDIO_NAME)[0],orig_path,processed_path,str(os.getlogin()),time.strftime("%Y-%m-%d"),time.strftime("%H:%M:%S")))
            
            # connection is not autocommit by default. So you must commit to save your changes.
            # connection.commit()
            audio_id_1=cursor.lastrowid
        
        #second: store result data to TBL_STT_RESULT table
        #from dataframe"output" to get the SD, STT and label results
            # Create a new record
            sql2 = "INSERT INTO `TBL_STT_RESULT` (`audio_id`, `slice_id`,`start_time`,`end_time`,`duration`,`text`,`speaker_label`,`save_path`,`create_by`,`create_date`,`create_time`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            for i in range(0,len(resultDf)-1):
                cursor.execute(sql2, (audio_id_1,resultDf.index[i], resultDf.starttime[i],resultDf.endtime[i],resultDf.duration[i],resultDf.text[i],resultDf.label[i],slices_path,str(os.getlogin()),time.strftime("%Y-%m-%d"),time.strftime("%H:%M:%S")))

        # connection is not autocommit by default. So you must commit to save your changes.
        connection.commit()
