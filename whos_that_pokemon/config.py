import os

SEED = 42

OUT_SIZE = 128
N_BG_PER_POKEMON = 6
VAL_FRACTION = 0.15
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

CLS_BATCH_SIZE = 32
CLS_EPOCHS = 20
CLS_LR = 1e-3

RAW_DIR = "raw_sprites"
IMG_DIR = "dataset/images"
MASK_DIR = "dataset/masks"

SPRITE_BASE_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork"
NAMES_CSV_URL = "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/pokemon.csv"

SEG_CHECKPOINT = "best_segmentation_model.pt"
CLS_CHECKPOINT = "best_classifier_model.pt"

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)