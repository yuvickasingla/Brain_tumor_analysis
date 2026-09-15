import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOT = os.path.join(BASE_DIR, 'data', 'brisc2025')

BRISC_TRAIN_DIR = os.path.join(DATASET_ROOT, 'classification_task', 'train')
BRISC_TEST_DIR  = os.path.join(DATASET_ROOT, 'classification_task', 'test')

DATA_DIR   = os.path.join(BASE_DIR, 'data', 'split')
TRAIN_DIR  = os.path.join(DATA_DIR, 'train')
VAL_DIR    = os.path.join(DATA_DIR, 'val')
TEST_DIR   = os.path.join(DATA_DIR, 'test')

MODEL_DIR       = os.path.join(BASE_DIR, 'models')
LOGS_DIR        = os.path.join(BASE_DIR, 'logs')
OUTPUTS_DIR     = os.path.join(BASE_DIR, 'outputs')

for _d in [MODEL_DIR, LOGS_DIR, OUTPUTS_DIR]:
    os.makedirs(_d, exist_ok=True)

def model_save_path(name: str) -> str:
    """Return checkpoint path for a given model name."""
    return os.path.join(MODEL_DIR, f'{name}_best.pth')
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE   = 224          
CHANNELS     = 3
BATCH_SIZE   = 32
NUM_EPOCHS   = 25
SEED         = 42

LR = {
    'custom_cnn':    1e-4,
    'resnet50':      1e-4,   
    'efficientnet':  1e-4,
    'mobilenetv3':   1e-4,
}
LR_BACKBONE = 1e-5         

VAL_SPLIT    = 0.20       
PATIENCE     = 5            

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
