from pathlib import Path


# =====================================================
# MODEL
# =====================================================

MODEL_NAME = "jazzmacedo/fruits-and-vegetables-detector-36"


# =====================================================
# CESTY
# =====================================================

INPUT_PATH = Path("videos/banana.mp4")
OUTPUT_DIR = Path("output/selected_frames")
ROI_VIDEO_PATH = Path("output/roi.mp4")


# =====================================================
# SPRACOVANIE VIDEA
# =====================================================

# Začiatok kontroly videa v sekundách.
START_TIME = 4.0

# Doba, počas ktorej musí byť scéna stabilná.
STABLE_TIME = 0.5

# Ak sa zmení viac ako toto percento pixelov, považujeme to za pohyb.
MOTION_THRESHOLD = 0.02

# Minimálna ostrosť obrazu.
MIN_SHARPNESS = 10

# Počet kandidátov a počet vybraných framov.
CANDIDATE_COUNT = 6
SELECTED_COUNT = 3

# ROI ovocia: x1, y1, x2, y2.
ROI = (620, 40, 1420, 650)

# Parametre detekcie pohybu.
MOTION_RESIZE = (320, 240)
MOTION_BLUR_KERNEL = (5, 5)
MOTION_DIFF_THRESHOLD = 25

# Náhradná hodnota, ak video neposkytne platné FPS.
DEFAULT_FPS = 30.0

# Codec pre výstupné ROI video.
VIDEO_CODEC = "mp4v"
