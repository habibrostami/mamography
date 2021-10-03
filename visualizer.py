import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from keras import models
import matplotlib.pyplot as plt
from MaskRCNN.config import Config
from MaskRCNN import utils
from MaskRCNN import model as modellib
from MaskRCNN import visualize
from MammoDataset import *
from config_mrcnn import *
from sklearn import metrics
import itertools
import keras.preprocessing.image as Kimage
import fnmatch
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
import csv
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
from config import *
#from keras.models import load_model
from keras import models
if sys.platform == 'win32':
    import winsound
from quiver_engine import server

config = MammoInferenceConfig(DATASET_PATH)
model_mrcnn = modellib.MaskRCNN(mode="inference", config=config, model_dir='./logs')
server.launch(model_mrcnn,
              input_folder=r'D:\Master of Science\Datasets\INBreast\PNG-Dataset-v5\final - 1024 cheating mrcnn\Test\mammo')
