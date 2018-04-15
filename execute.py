# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:33:26 2018

@author: mstei
"""

import subprocess
import sys

theproc=subprocess.Popen([sys.executable, './drive.py', 'model.h5'])
theproc.communicate()