import sys
import os
import queue
import math
from pynput import keyboard

import MFCC
from sklearn.cluster import KMeans
import hmmlearn.hmm
import numpy as np
import pickle as pk

import librosa
import sounddevice as sd
import soundfile as sf

