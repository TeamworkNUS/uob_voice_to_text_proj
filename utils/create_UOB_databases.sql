-- # show databases;

-- show databases UOBtests;
-- # show databases like 'UOBtests';

-- ##for method 1：
-- # show engines;
-- # drop table `UOBtests`.`TBL_AUDIO`;
-- # drop table `UOBtests`.`TBL_STT_RESULT`;


CREATE TABLE IF NOT EXISTS `UOBtests`.`TBL_AUDIO`(
                                       `audio_id` int NOT NULL AUTO_INCREMENT,
                                       `audio_name` varchar(50)  NOT NULL,
                                       `path_orig` varchar(100)  NOT NULL,
                                       `path_processed` varchar(100) NOT NULL,
                                       `create_by` varchar(50) NOT NULL,
                                       `create_date` varchar(20) NOT NULL,
                                       `create_time` time NOT NULL,
                                       PRIMARY KEY (`audio_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin AUTO_INCREMENT=1 ;
-- #Specify the initial value of the self-increment: 1
-- #A legacy of mysql, mysql utf8 can only support up to 3bytes length of character encoding, for some need to occupy 4bytes of text, mysql utf8 is not supported, you have to use utf8mb4
-- #mb4 is most bytes 4, which uses 4 bytes to represent the full UTF-8
-- #utf8mb4_bin: compile and store each character of the string in binary data, case sensitive, and can store the binary content

desc `UOBtests`.`TBL_AUDIO`;
-- select * from `UOBtests`.`TBL_AUDIO`;



CREATE TABLE IF NOT EXISTS `UOBtests`.`TBL_STT_RESULT`(
                                    `audio_id` int NOT NULL,
                                    `slice_id` int NOT NULL,
                                    `start_time` float NOT NULL,
                                    `end_time` float NOT NULL,
                                    `duration` float NOT NULL,
                                    `speaker_label` varchar(20)  NOT NULL,
                                    `text` varchar(5000),
                                    `save_path` varchar(100) NOT NULL,
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
