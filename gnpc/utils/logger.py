import datetime
import logging
import warnings
import sys
import os
from logging import FileHandler

logging.captureWarnings(True)


class LogMsg:
    def __init__(
        self,
        fpath_log,
        log_name="MyLog",
        is_append: bool = False,
        include_datetime: bool = True,
    ):
        """
        Logging

        """
        self.fpath_log = fpath_log
        self.log_name = log_name
        self.is_append = is_append
        if self.is_append:
            self.mode = "a"
        else:
            self.mode = "w"
        self.include_datetime = include_datetime

        self.logger = self.get_logger()

    @staticmethod
    def convert_time(seconds):
        return str(datetime.timedelta(seconds=seconds))

    def is_log_exist(self):
        if os.path.exists(self.fpath_log):
            self.msg("\nAPPENDING PREVIOUS LOG\n")

    def get_formatter(self):
        if self.include_datetime:
            return logging.Formatter("%(asctime)s — %(message)s")
        else:
            return logging.Formatter("%(message)s")

    def get_console_out(self):
        formatter = self.get_formatter()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        return console_handler

    def get_console_err(self):
        formatter = self.get_formatter()
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        return console_handler

    def get_file_handler(self):
        formatter = self.get_formatter()
        file_handler = FileHandler(filename=self.fpath_log, mode=self.mode)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        return file_handler

    def get_logger(self):
        dir_name = os.path.dirname(self.fpath_log)
        if not os.path.exists(dir_name) and len(dir_name) > 0:
            os.makedirs(dir_name)

        logger = logging.getLogger(self.log_name)
        logger.setLevel(logging.INFO)

        logger.propagate = False
        # THIS IS CRUCIAL FOR DUPLICATE AVOIDANCE
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(self.get_console_out())
        # logger.addHandler(self.get_console_err())
        logger.addHandler(self.get_file_handler())

        return logger

    def msg(self, message: str):
        message = str(message)
        self.logger.info(message)


if __name__ == "__main__":

    # sample usage
    # "w" for replace existing log
    # "a" for append log
    log = LogMsg(fpath_log="log.log", log_name="mylog")
    log.msg("some log")
