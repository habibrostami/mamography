import os
# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Path to trained weights file
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

DATASET_PATH = r'D:\Master of Science\Datasets\INBreast\PNG-Dataset-v5\final - 1024 cheating mrcnn'

#CLASSIFY_DATASET_PATH = r'D:\Master of Science\Datasets\INBreast\PNG-Dataset-v5\Patch-300rnd-v200716\Test'
CLASSIFY_DATASET_PATH = r'D:\Master of Science\Datasets\INBreast\PNG-Dataset-v5\final - 1024 cheating mrcnn\Test'

CNN_MODEL = 'Xception'
MRCNN_WEIGHTS = 'mrcnn_inbreast-resnet101'