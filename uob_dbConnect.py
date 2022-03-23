import pymysql.cursors
from init import (
    dbName,
    dbPwd
)

def connectDB():
    connection= pymysql.connect(host='localhost',
                            user='root',
                            password= dbPwd,#Here fill in the password set up by the mysql database administrator
                            database= dbName,
                            charset='utf8mb4',
                            cursorclass=pymysql.cursors.DictCursor)
    return connection