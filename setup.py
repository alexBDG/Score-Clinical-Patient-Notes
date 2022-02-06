# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:24:19 2022

@author: Shion Pavlichenko
"""
import subprocess
import sys

package = "requirements.txt"

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r",package])