-- # show databases;

-- show databases UOBtests;
-- # show databases like 'UOBtests';

-- ##for method 1：
-- # show engines;
-- # drop table `UOBtests`.`TBL_AUDIO`;
-- # drop table `UOBtests`.`TBL_STT_RESULT`;


-- -----------------------------------------------------------------------------------
-- |                                      Tables                                     |
-- -----------------------------------------------------------------------------------
use UOBtests;
drop table `UOBtests`.`TBL_STT_RESULT`;
drop table `UOBtests`.`TBL_AUDIO`;
drop table `UOBtests`.`TBL_VERSIONS`;
drop table `UOBtests`.`TBL_PROCESS_LOG`;
drop procedure `UOBtests`.`Initialize_TBL_VERSIONS`;

CREATE TABLE IF NOT EXISTS `UOBtests`.`TBL_AUDIO`(
                                       `audio_id` varchar(30) NOT NULL, -- int NOT NULL AUTO_INCREMENT,
                                       `audio_name` varchar(50) NOT NULL,
                                       `path_orig` varchar(100) NOT NULL,
                                       `audio_name_processed` varchar(50),
                                       `path_processed` varchar(100),
                                       `upload_filename` varchar(50) NOT NULL,
                                       `path_upload` varchar(100) NOT NULL,
                                       `upload_file_count` int NOT NULL default 1,
                                       `description` varchar(200),
                                       `audio_meta` varchar(200) NOT NULL,
                                       `analysis` varchar(200), -- 
                                       `flg_delete` varchar(1) default Null, -- 'X' means deletion; Null means active.
                                       `create_by` varchar(50) NOT NULL,
                                       `create_date` varchar(20) NOT NULL,
                                       `create_time` time NOT NULL,
                                       `update_by` varchar(50) NOT NULL,
                                       `update_date` varchar(20) NOT NULL,
                                       `update_time` time NOT NULL,
                                       PRIMARY KEY (`audio_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin AUTO_INCREMENT=1 ;
-- #Specify the initial value of the self-increment: 1
-- #A legacy of mysql, mysql utf8 can only support up to 3bytes length of character encoding, for some need to occupy 4bytes of text, mysql utf8 is not supported, you have to use utf8mb4
-- #mb4 is most bytes 4, which uses 4 bytes to represent the full UTF-8
-- #utf8mb4_bin: compile and store each character of the string in binary data, case sensitive, and can store the binary content

desc `UOBtests`.`TBL_AUDIO`;
-- select * from `UOBtests`.`TBL_AUDIO`;



CREATE TABLE IF NOT EXISTS `UOBtests`.`TBL_STT_RESULT`(
                                    `audio_id` varchar(30) NOT NULL,
                                    `slice_id` int NOT NULL,
                                    `start_time` float NOT NULL,
                                    `end_time` float NOT NULL,
                                    `duration` float NOT NULL,
                                    `speaker_label` varchar(20) NOT NULL,
                                    `text` varchar(5000),
                                    `slice_name` varchar(50) NOT NULL,
                                    `slice_path` varchar(100) NOT NULL,
                                    `create_by` varchar(50) NOT NULL,
                                    `create_date` varchar(20)  NOT NULL,
                                    `create_time` time NOT NULL,
                                    PRIMARY KEY (`audio_id`,`slice_id`),
                                    FOREIGN KEY (`audio_id`)
                                        REFERENCES `TBL_AUDIO` (`audio_id`)
                                                ON UPDATE CASCADE
                                                ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
--     #A legacy of mysql, mysql utf8 can only support up to 3bytes length of character encoding, for some need to occupy 4bytes of text, mysql utf8 is not supported, you have to use utf8mb4
--     #mb4 is most bytes 4, which uses 4 bytes to represent the full UTF-8
--     #utf8mb4_bin: compile and store each character of the string in binary data, case sensitive, and can store the binary content
-- #   AUTO_INCREMENT=1 ;#Specify the initial value of the self-increment: 1



desc `UOBtests`.`TBL_STT_RESULT`;
-- select * from `UOBtests`.`TBL_STT_RESULT`;


CREATE TABLE IF NOT EXISTS `UOBtests`.`TBL_VERSIONS`(
    `version_id` int NOT NULL AUTO_INCREMENT,
    `version_name` varchar(30) NOT NULL,
    `version_value` varchar(30),
    PRIMARY KEY(`version_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

desc `UOBtests`.`TBL_VERSIONS`;



CREATE TABLE IF NOT EXISTS `UOBtests`.`TBL_PROCESS_LOG`(
    `log_id` varchar(30) NOT NULL,
    `audio_id` varchar(30) NOT NULL, 
    `params` varchar(200), -- e.g.{"NR":True, "SE":True, "SR":False, "SD":"resemblyzer", "STT":"vosk"}
    `analysis_name` varchar(200), -- e.g.{0:"SD", 1:"STT", 2:"UseCase1", 3:"UseCase2"}
    `message` varchar(200), -- e.g.{0:xxxx, 1:xxxxxx, ...}
    `process_time` varchar(200), -- e.g.{"starttime":"20220401 15:40:01", "endtime":"20220401 15:57:21", "duration":"00:17:20"}
    `create_by` varchar(50) NOT NULL,
    `create_on` date NOT NULL,
    PRIMARY KEY(`log_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

desc `UOBtests`.`TBL_PROCESS_LOG`;



-- -----------------------------------------------------------------------------------
-- |                                Stored Procedures                                |
-- -----------------------------------------------------------------------------------


CREATE PROCEDURE Initialize_TBL_VERSIONS()
BEGIN
    TRUNCATE TABLE TBL_VERSIONS;

    INSERT INTO `TBL_VERSIONS` (`VERSION_NAME`,`VERSION_VALUE`)
    VALUES ('AUDIO_ID_VER',1);

    SELECT * FROM TBL_VERSIONS;
END;



CALL Initialize_TBL_VERSIONS();