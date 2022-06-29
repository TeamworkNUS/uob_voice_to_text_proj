
CREATE DATABASE uobweb;
USE uobweb;
drop table `uobweb`.`analysis_sttresult`;
drop table `uobweb`.`analysis_audio`;
drop table `uobweb`.`analysis_versions`;
drop table `uobweb`.`analysis_process_log`;
drop table `uobweb`.`analysis_analysisselection`;
drop table `uobweb`.`analysis_personalinfo`;
drop procedure `uobweb`.`Initialize_analysis_versions`;

CREATE TABLE IF NOT EXISTS `analysis_audio` (
  `audio_id` varchar(30) NOT NULL,
  `audio_name` varchar(100) DEFAULT NULL,
  `path_orig` varchar(150) DEFAULT NULL,
  `audio_name_processed` varchar(100) DEFAULT NULL,
  `path_processed` varchar(150) DEFAULT NULL,
  `upload_filename` varchar(100) DEFAULT NULL,
  `upload_file_count` int NOT NULL,
  `description` longtext,
  `audio_meta` varchar(200) DEFAULT NULL,
  `create_by` varchar(50) NOT NULL,
  `create_date` date NOT NULL,
  `create_time` time(6) NOT NULL,
  `update_by` varchar(50) NOT NULL,
  `update_date` date NOT NULL,
  `update_time` time(6) NOT NULL,
  `path_upload` varchar(150) DEFAULT NULL,
  `analysis` varchar(200) DEFAULT NULL,
  `flg_delete` varchar(1) DEFAULT NULL,
  PRIMARY KEY (`audio_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;


 CREATE TABLE IF NOT EXISTS `analysis_sttresult` (
  `audio_slice_id` varchar(30) COLLATE utf8mb4_bin NOT NULL,
  `audio_id` varchar(30) COLLATE utf8mb4_bin NOT NULL,
  `slice_id` int NOT NULL,
  `start_time` float NOT NULL,
  `end_time` float NOT NULL,
  `duration` float NOT NULL,
  `speaker_label` varchar(20) COLLATE utf8mb4_bin NOT NULL,
  `text` varchar(5000) COLLATE utf8mb4_bin DEFAULT NULL,
  `slice_name` varchar(100) COLLATE utf8mb4_bin NOT NULL,
  `slice_path` varchar(150) COLLATE utf8mb4_bin NOT NULL,
  `create_by` varchar(50) COLLATE utf8mb4_bin NOT NULL,
  `create_date` varchar(20) COLLATE utf8mb4_bin NOT NULL,
  `create_time` time NOT NULL,
  PRIMARY KEY (`audio_slice_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;


CREATE TABLE IF NOT EXISTS `analysis_version` (
  `version_id` int NOT NULL AUTO_INCREMENT,
  `version_name` varchar(30) NOT NULL,
  `version_value` varchar(30) NOT NULL,
  PRIMARY KEY (`version_id`),
  UNIQUE KEY `version_name` (`version_name`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb3;


CREATE TABLE IF NOT EXISTS `analysis_process_log` (
  `log_id` varchar(30) COLLATE utf8mb4_bin NOT NULL,
  `audio_id` varchar(30) COLLATE utf8mb4_bin NOT NULL,
  `params` varchar(200) COLLATE utf8mb4_bin DEFAULT NULL,
  `analysis_name` varchar(200) COLLATE utf8mb4_bin DEFAULT NULL,
  `message` varchar(200) COLLATE utf8mb4_bin DEFAULT NULL,
  `process_time` varchar(200) COLLATE utf8mb4_bin DEFAULT NULL,
  `create_by` varchar(50) COLLATE utf8mb4_bin NOT NULL,
  `create_on` date NOT NULL,
  PRIMARY KEY (`log_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;


CREATE TABLE IF NOT EXISTS `analysis_analysisselection` (
  `analysisSelection_id` int NOT NULL AUTO_INCREMENT,
  `analysis_name` varchar(50) COLLATE utf8mb4_bin NOT NULL DEFAULT 'error',
  PRIMARY KEY (`analysisSelection_id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;


 CREATE TABLE IF NOT EXISTS `analysis_personalinfo` (
  `audio_slice_id` varchar(30) COLLATE utf8mb4_bin NOT NULL,
  `audio_id` varchar(30) COLLATE utf8mb4_bin NOT NULL,
  `slice_id` int NOT NULL,
  `is_kyc` varchar(5) COLLATE utf8mb4_bin DEFAULT NULL,
  `is_pii` varchar(5) COLLATE utf8mb4_bin DEFAULT NULL,
  `create_by` varchar(50) COLLATE utf8mb4_bin NOT NULL,
  `create_date` varchar(20) COLLATE utf8mb4_bin NOT NULL,
  `create_time` time NOT NULL,
  PRIMARY KEY (`audio_slice_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
-- -----------------------------------------------------------------------------------
-- |                                Stored Procedures                                |
-- -----------------------------------------------------------------------------------


CREATE PROCEDURE Initialize_analysis_version()
BEGIN
    TRUNCATE TABLE analysis_version;

    INSERT INTO `analysis_version` (`VERSION_NAME`,`VERSION_VALUE`)
    VALUES ('AUDIO_ID_VER',1);
    INSERT INTO `analysis_version` (`VERSION_NAME`,`VERSION_VALUE`)
    VALUES ('UPLOAD_ID_VER',1);


    SELECT * FROM analysis_version;
END;



-- -----------------------------------------------------------------------------------
-- |                                 Initialization                                  |
-- -----------------------------------------------------------------------------------
--  analysis_version
CALL Initialize_analysis_version();
--  analysis selection

INSERT INTO `uobweb`.`analysis_analysisselection` (`analysis_name`)
     VALUES ('SD+STT'),
            ('KYC+PII');

