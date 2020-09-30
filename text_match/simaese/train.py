import os
import sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import tensorflow as tf
from utils.load_data import load_char_data
from text_match.simaese.graph import Graph
from text_match.simaese import args
