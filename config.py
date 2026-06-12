# ======================
# PATHS
# ======================

MODEL_PATH = "models/hand_number_model.keras"

DATASET_PATH = "dataset"

HAND_MODEL_PATH = "hand_landmarker.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-assets/"
    "hand_landmarker.task"
)


# ======================
# TRAINING
# ======================
HAND_NUMBER = 2
# ======================
# CAMERA
# ======================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ======================
# CNN MODEL
# ======================
INPUT_SIZE = 224
NUM_CLASSES = 6
EPOCHS = 30
BATCH_SIZE = 32

TARGET_IMAGES = 500
# ======================
# HAND DETECTION (MediaPipe)
# ======================

MIN_HAND_DETECTION_CONFIDENCE = 0.5

MIN_HAND_PRESENCE_CONFIDENCE = 0.5

MIN_TRACKING_CONFIDENCE = 0.5

# ======================
# DATA COLLECTION
# ======================

SAVE_SIZE = (
    INPUT_SIZE,
    INPUT_SIZE
)

# ======================
# DETECTION
# ======================

SMOOTHING_WINDOW = 10

MIN_CONFIDENCE = 0.40

# ======================
# CLASSES
# ======================

CLASS_NAMES = [
    "0", "1", "2", "3", "4",
    "5"
]