import os

DATA_DIR = os.environ.get("CHESTXRAY_DATA_DIR", "./data")
CSV_PATH = os.path.join(DATA_DIR, "Data_Entry_2017.csv")
TRAIN_LIST = os.path.join(DATA_DIR, "train_val_list.txt")
TEST_LIST = os.path.join(DATA_DIR, "test_list.txt")
IMAGE_DIR = DATA_DIR

NUM_CLASSES = 14
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-4
WEIGHT_DECAY = 1e-4
BETA = 0.9999
GAMMA = 2.0
NUM_WORKERS = 2
SEED = 42
VALIDATION_FRACTION = 0.1

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural Thickening",
    "Pneumonia", "Pneumothorax"
]

CHECKPOINT = "pulmoNeXt_best.pth"
