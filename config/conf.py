# -*- coding utf-8 -*-

import os
from datetime import datetime


class Settings:
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, '..'))
    SRC_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    config = None

    def __init__(self):
        self.logger = self.get_logger()

    def get_logger(self):
        logger = 1 #create logger tbd
        return logger


