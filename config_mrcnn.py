import os
import sys
# Import Mask RCNN
#sys.path.append(os.path.abspath("./Mask_RCNN"))  # To find local version of the library
from MaskRCNN.config import Config
import numpy as np

# define a configuration for the model
class MammoTrainingConfig(Config):
    # define the name of the configuration
    NAME = "mammo"
    REDUCE_LR = False
    TRAINING_MONITOR = "loss"
    GPU_COUNT = 1
    # number of classes (BG + Mass + Calc)
    NUM_CLASSES = 3
    IMAGES_PER_GPU = 1
    #BACKBONE = 'resnet50'
    BACKBONE = 'resnet101'
    TRAIN_ROIS_PER_IMAGE = 512
    LEARNING_RATE = 0.001
    IMAGE_RESIZE_MODE = "square"
    #IMAGE_MIN_DIM = 14*64
    IMAGE_MAX_DIM = 14*64
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    USE_MINI_MASK = False
    IMAGE_MIN_SCALE = 0
    IMAGE_CHANNEL_COUNT = 3
    RPN_NMS_THRESHOLD = 0.9
    # Minimum probability value to accept a detected instance
    DETECTION_MIN_CONFIDENCE = 0.9
    OPTIMIZER = "sgd"
    TRAIN_BN = False
    POST_NMS_ROIS_INFERENCE = 100

    ## Resolution
    #RES_FACTOR = 1
    #IMAGE_MAX_DIM = 1024 // RES_FACTOR
    #RPN_ANCHOR_SCALES = tuple(np.divide((32, 64, 128, 256, 512), RES_FACTOR))


    def __init__(self, dataset_dir, learning_rate=None):
        train_path = os.path.join(dataset_dir, 'Train', 'mammo')
        train_len = len([name for name in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, name))])
        self.STEPS_PER_EPOCH = train_len / self.IMAGES_PER_GPU
        val_path = os.path.join(dataset_dir, 'Val', 'mammo')
        val_len = len([name for name in os.listdir(val_path) if os.path.isfile(os.path.join(val_path, name))])
        self.VALIDATION_STEPS = val_len / self.IMAGES_PER_GPU
        if learning_rate is not None:
            self.LEARNING_RATE = float(learning_rate)
        super().__init__()


class MammoInferenceConfig(MammoTrainingConfig):
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100
    DETECTION_MIN_CONFIDENCE = 0.8
    TRAIN_BN = False
    #BACKBONE = 'resnet50'
    BACKBONE = 'resnet101'