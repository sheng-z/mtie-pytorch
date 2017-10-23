#!/usr/bin/env python
# encoding: utf-8


import sys
import random
import logging
import numpy as np
import torch
import datetime


class ProgressBar(object):
    def __init__(self, progress_name, input_file=None, total_steps=None, progressbar_width=100):
        """
        If the progress bar is line-count based, just pass in the file and we'll use
        the line count of the file as the total steps. Otherwise you must pass in the total_steps
        param.
        """
        self.progress_name = progress_name
        if input_file is None and total_steps is None:
            raise ValueError("Either input_file or total_steps must be specified.")
        if input_file is not None and total_steps is not None:
            raise ValueError("Only one of input_file or total_steps can be specified.")

        self.progressbar_width = progressbar_width
        self.total_steps = total_steps if total_steps is not None else ProgressBar.get_line_count(input_file)
        self.progressbar_unit_current_line = 0
        self.progressbar_unit = self.total_steps / self.progressbar_width
        self.start_ts = None

    @staticmethod
    def get_line_count(input_file):
        import subprocess
        p = subprocess.Popen(['wc', '-l', input_file], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        result, err = p.communicate()
        if p.returncode != 0:
            raise IOError(err)
        return int(result.strip().split()[0])

    def start_progress_bar(self):
        self.start_ts = datetime.datetime.now()
        print "%s started..." % self.progress_name
        sys.stdout.write("[%s]" % (" " * self.progressbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.progressbar_width + 1))

    def increment_step(self):
        self.progressbar_unit_current_line += 1
        if self.progressbar_unit_current_line >= self.progressbar_unit:
            sys.stdout.write("-")
            sys.stdout.flush()
            self.progressbar_unit_current_line = 0

    def stop_progress_bar(self):
        # Move the cursor to a newline to avoid override the current progress bar.
        print
        print "%s completed! It took %s." % (self.progress_name, datetime.datetime.now() - self.start_ts)
        print


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logging(level=logging.DEBUG):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log

